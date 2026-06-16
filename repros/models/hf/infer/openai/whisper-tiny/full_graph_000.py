class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "bf16[1, 80, 3000][240000, 3000, 1]cuda:0", arg1_1: "bf16[384, 80, 3][240, 3, 1]cuda:0", arg2_1: "bf16[384][1]cuda:0", arg3_1: "bf16[384, 384, 3][1152, 3, 1]cuda:0", arg4_1: "bf16[384][1]cuda:0", arg5_1: "bf16[1500, 384][384, 1]cuda:0", arg6_1: "bf16[384][1]cuda:0", arg7_1: "bf16[384][1]cuda:0", arg8_1: "bf16[384, 384][384, 1]cuda:0", arg9_1: "bf16[384][1]cuda:0", arg10_1: "bf16[384, 384][384, 1]cuda:0", arg11_1: "bf16[384, 384][384, 1]cuda:0", arg12_1: "bf16[384][1]cuda:0", arg13_1: "bf16[384, 384][384, 1]cuda:0", arg14_1: "bf16[384][1]cuda:0", arg15_1: "bf16[384][1]cuda:0", arg16_1: "bf16[384][1]cuda:0", arg17_1: "bf16[1536, 384][384, 1]cuda:0", arg18_1: "bf16[1536][1]cuda:0", arg19_1: "bf16[384, 1536][1536, 1]cuda:0", arg20_1: "bf16[384][1]cuda:0", arg21_1: "bf16[384][1]cuda:0", arg22_1: "bf16[384][1]cuda:0", arg23_1: "bf16[384, 384][384, 1]cuda:0", arg24_1: "bf16[384][1]cuda:0", arg25_1: "bf16[384, 384][384, 1]cuda:0", arg26_1: "bf16[384, 384][384, 1]cuda:0", arg27_1: "bf16[384][1]cuda:0", arg28_1: "bf16[384, 384][384, 1]cuda:0", arg29_1: "bf16[384][1]cuda:0", arg30_1: "bf16[384][1]cuda:0", arg31_1: "bf16[384][1]cuda:0", arg32_1: "bf16[1536, 384][384, 1]cuda:0", arg33_1: "bf16[1536][1]cuda:0", arg34_1: "bf16[384, 1536][1536, 1]cuda:0", arg35_1: "bf16[384][1]cuda:0", arg36_1: "bf16[384][1]cuda:0", arg37_1: "bf16[384][1]cuda:0", arg38_1: "bf16[384, 384][384, 1]cuda:0", arg39_1: "bf16[384][1]cuda:0", arg40_1: "bf16[384, 384][384, 1]cuda:0", arg41_1: "bf16[384, 384][384, 1]cuda:0", arg42_1: "bf16[384][1]cuda:0", arg43_1: "bf16[384, 384][384, 1]cuda:0", arg44_1: "bf16[384][1]cuda:0", arg45_1: "bf16[384][1]cuda:0", arg46_1: "bf16[384][1]cuda:0", arg47_1: "bf16[1536, 384][384, 1]cuda:0", arg48_1: "bf16[1536][1]cuda:0", arg49_1: "bf16[384, 1536][1536, 1]cuda:0", arg50_1: "bf16[384][1]cuda:0", arg51_1: "bf16[384][1]cuda:0", arg52_1: "bf16[384][1]cuda:0", arg53_1: "bf16[384, 384][384, 1]cuda:0", arg54_1: "bf16[384][1]cuda:0", arg55_1: "bf16[384, 384][384, 1]cuda:0", arg56_1: "bf16[384, 384][384, 1]cuda:0", arg57_1: "bf16[384][1]cuda:0", arg58_1: "bf16[384, 384][384, 1]cuda:0", arg59_1: "bf16[384][1]cuda:0", arg60_1: "bf16[384][1]cuda:0", arg61_1: "bf16[384][1]cuda:0", arg62_1: "bf16[1536, 384][384, 1]cuda:0", arg63_1: "bf16[1536][1]cuda:0", arg64_1: "bf16[384, 1536][1536, 1]cuda:0", arg65_1: "bf16[384][1]cuda:0", arg66_1: "bf16[384][1]cuda:0", arg67_1: "bf16[384][1]cuda:0", arg68_1: "i64[1, 1][1, 1]cuda:0", arg69_1: "bf16[51865, 384][384, 1]cuda:0", arg70_1: "bf16[448, 384][384, 1]cuda:0", arg71_1: "bf16[384][1]cuda:0", arg72_1: "bf16[384][1]cuda:0", arg73_1: "bf16[384, 384][384, 1]cuda:0", arg74_1: "bf16[384][1]cuda:0", arg75_1: "bf16[384, 384][384, 1]cuda:0", arg76_1: "bf16[384, 384][384, 1]cuda:0", arg77_1: "bf16[384][1]cuda:0", arg78_1: "bf16[384, 384][384, 1]cuda:0", arg79_1: "bf16[384][1]cuda:0", arg80_1: "bf16[384][1]cuda:0", arg81_1: "bf16[384][1]cuda:0", arg82_1: "bf16[384, 384][384, 1]cuda:0", arg83_1: "bf16[384][1]cuda:0", arg84_1: "bf16[384, 384][384, 1]cuda:0", arg85_1: "bf16[384, 384][384, 1]cuda:0", arg86_1: "bf16[384][1]cuda:0", arg87_1: "bf16[384, 384][384, 1]cuda:0", arg88_1: "bf16[384][1]cuda:0", arg89_1: "bf16[384][1]cuda:0", arg90_1: "bf16[384][1]cuda:0", arg91_1: "bf16[1536, 384][384, 1]cuda:0", arg92_1: "bf16[1536][1]cuda:0", arg93_1: "bf16[384, 1536][1536, 1]cuda:0", arg94_1: "bf16[384][1]cuda:0", arg95_1: "bf16[384][1]cuda:0", arg96_1: "bf16[384][1]cuda:0", arg97_1: "bf16[384, 384][384, 1]cuda:0", arg98_1: "bf16[384][1]cuda:0", arg99_1: "bf16[384, 384][384, 1]cuda:0", arg100_1: "bf16[384, 384][384, 1]cuda:0", arg101_1: "bf16[384][1]cuda:0", arg102_1: "bf16[384, 384][384, 1]cuda:0", arg103_1: "bf16[384][1]cuda:0", arg104_1: "bf16[384][1]cuda:0", arg105_1: "bf16[384][1]cuda:0", arg106_1: "bf16[384, 384][384, 1]cuda:0", arg107_1: "bf16[384][1]cuda:0", arg108_1: "bf16[384, 384][384, 1]cuda:0", arg109_1: "bf16[384, 384][384, 1]cuda:0", arg110_1: "bf16[384][1]cuda:0", arg111_1: "bf16[384, 384][384, 1]cuda:0", arg112_1: "bf16[384][1]cuda:0", arg113_1: "bf16[384][1]cuda:0", arg114_1: "bf16[384][1]cuda:0", arg115_1: "bf16[1536, 384][384, 1]cuda:0", arg116_1: "bf16[1536][1]cuda:0", arg117_1: "bf16[384, 1536][1536, 1]cuda:0", arg118_1: "bf16[384][1]cuda:0", arg119_1: "bf16[384][1]cuda:0", arg120_1: "bf16[384][1]cuda:0", arg121_1: "bf16[384, 384][384, 1]cuda:0", arg122_1: "bf16[384][1]cuda:0", arg123_1: "bf16[384, 384][384, 1]cuda:0", arg124_1: "bf16[384, 384][384, 1]cuda:0", arg125_1: "bf16[384][1]cuda:0", arg126_1: "bf16[384, 384][384, 1]cuda:0", arg127_1: "bf16[384][1]cuda:0", arg128_1: "bf16[384][1]cuda:0", arg129_1: "bf16[384][1]cuda:0", arg130_1: "bf16[384, 384][384, 1]cuda:0", arg131_1: "bf16[384][1]cuda:0", arg132_1: "bf16[384, 384][384, 1]cuda:0", arg133_1: "bf16[384, 384][384, 1]cuda:0", arg134_1: "bf16[384][1]cuda:0", arg135_1: "bf16[384, 384][384, 1]cuda:0", arg136_1: "bf16[384][1]cuda:0", arg137_1: "bf16[384][1]cuda:0", arg138_1: "bf16[384][1]cuda:0", arg139_1: "bf16[1536, 384][384, 1]cuda:0", arg140_1: "bf16[1536][1]cuda:0", arg141_1: "bf16[384, 1536][1536, 1]cuda:0", arg142_1: "bf16[384][1]cuda:0", arg143_1: "bf16[384][1]cuda:0", arg144_1: "bf16[384][1]cuda:0", arg145_1: "bf16[384, 384][384, 1]cuda:0", arg146_1: "bf16[384][1]cuda:0", arg147_1: "bf16[384, 384][384, 1]cuda:0", arg148_1: "bf16[384, 384][384, 1]cuda:0", arg149_1: "bf16[384][1]cuda:0", arg150_1: "bf16[384, 384][384, 1]cuda:0", arg151_1: "bf16[384][1]cuda:0", arg152_1: "bf16[384][1]cuda:0", arg153_1: "bf16[384][1]cuda:0", arg154_1: "bf16[384, 384][384, 1]cuda:0", arg155_1: "bf16[384][1]cuda:0", arg156_1: "bf16[384, 384][384, 1]cuda:0", arg157_1: "bf16[384, 384][384, 1]cuda:0", arg158_1: "bf16[384][1]cuda:0", arg159_1: "bf16[384, 384][384, 1]cuda:0", arg160_1: "bf16[384][1]cuda:0", arg161_1: "bf16[384][1]cuda:0", arg162_1: "bf16[384][1]cuda:0", arg163_1: "bf16[1536, 384][384, 1]cuda:0", arg164_1: "bf16[1536][1]cuda:0", arg165_1: "bf16[384, 1536][1536, 1]cuda:0", arg166_1: "bf16[384][1]cuda:0", arg167_1: "bf16[384][1]cuda:0", arg168_1: "bf16[384][1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:618 in forward, code: inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        convolution: "bf16[1, 384, 3000][1152000, 3000, 1]cuda:0" = torch.ops.aten.convolution.default(arg0_1, arg1_1, arg2_1, [1], [1], [1], False, [0], 1);  arg0_1 = arg1_1 = arg2_1 = None
        convert_element_type: "f32[1, 384, 3000][1152000, 3000, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution, torch.float32);  convolution = None
        mul: "f32[1, 384, 3000][1152000, 3000, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type, 0.5)
        mul_1: "f32[1, 384, 3000][1152000, 3000, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type, 0.7071067811865476);  convert_element_type = None
        erf: "f32[1, 384, 3000][1152000, 3000, 1]cuda:0" = torch.ops.aten.erf.default(mul_1);  mul_1 = None
        add: "f32[1, 384, 3000][1152000, 3000, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_2: "f32[1, 384, 3000][1152000, 3000, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, add);  mul = add = None
        convert_element_type_1: "bf16[1, 384, 3000][1152000, 3000, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:619 in forward, code: inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))
        convolution_1: "bf16[1, 384, 1500][576000, 1500, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_1, arg3_1, arg4_1, [2], [1], [1], False, [0], 1);  convert_element_type_1 = arg3_1 = arg4_1 = None
        convert_element_type_2: "f32[1, 384, 1500][576000, 1500, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_1, torch.float32);  convolution_1 = None
        mul_3: "f32[1, 384, 1500][576000, 1500, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2, 0.5)
        mul_4: "f32[1, 384, 1500][576000, 1500, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2, 0.7071067811865476);  convert_element_type_2 = None
        erf_1: "f32[1, 384, 1500][576000, 1500, 1]cuda:0" = torch.ops.aten.erf.default(mul_4);  mul_4 = None
        add_1: "f32[1, 384, 1500][576000, 1500, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_5: "f32[1, 384, 1500][576000, 1500, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3, add_1);  mul_3 = add_1 = None
        convert_element_type_3: "bf16[1, 384, 1500][576000, 1500, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:621 in forward, code: inputs_embeds = inputs_embeds.permute(0, 2, 1)
        permute: "bf16[1, 1500, 384][576000, 1, 1500]cuda:0" = torch.ops.aten.permute.default(convert_element_type_3, [0, 2, 1]);  convert_element_type_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:622 in forward, code: all_positions = torch.arange(self.embed_positions.num_embeddings, device=inputs_embeds.device)
        iota: "i64[1500][1]cuda:0" = torch.ops.prims.iota.default(1500, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:624 in forward, code: hidden_states = inputs_embeds + self.embed_positions(all_positions)
        embedding: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.embedding.default(arg5_1, iota);  arg5_1 = iota = None
        add_2: "bf16[1, 1500, 384][576000, 1, 1500]cuda:0" = torch.ops.aten.add.Tensor(permute, embedding);  permute = embedding = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:392 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        clone_1: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.clone.default(add_2, memory_format = torch.contiguous_format)
        convert_element_type_4: "f32[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_1, torch.float32);  clone_1 = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_4, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 1500, 1][1500, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[1, 1500, 1][1500, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        sub: "f32[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_4, getitem_1);  convert_element_type_4 = getitem_1 = None
        add_3: "f32[1, 1500, 1][1500, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[1, 1500, 1][1500, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        mul_6: "f32[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_7: "f32[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_6, arg6_1);  mul_6 = arg6_1 = None
        add_4: "f32[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_7, arg7_1);  mul_7 = arg7_1 = None
        convert_element_type_5: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_4, torch.bfloat16);  add_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        view: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_5, [1500, 384])
        permute_1: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg9_1, view, permute_1);  arg9_1 = view = permute_1 = None
        view_1: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm, [1, 1500, 384]);  addmm = None
        mul_8: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1, 0.125);  view_1 = None
        view_2: "bf16[1, 1500, 6, 64][576000, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_8, [1, 1500, -1, 64]);  mul_8 = None
        permute_2: "bf16[1, 6, 1500, 64][576000, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        clone_2: "bf16[1, 6, 1500, 64][576000, 96000, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_3: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_5, [1500, 384])
        permute_3: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        mm: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_3, permute_3);  view_3 = permute_3 = None
        view_4: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [1, 1500, 384]);  mm = None
        view_5: "bf16[1, 1500, 6, 64][576000, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_4, [1, -1, 6, 64]);  view_4 = None
        permute_4: "bf16[1, 6, 1500, 64][576000, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        clone_3: "bf16[1, 6, 1500, 64][576000, 96000, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_6: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_5, [1500, 384]);  convert_element_type_5 = None
        permute_5: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_1: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg12_1, view_6, permute_5);  arg12_1 = view_6 = permute_5 = None
        view_7: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_1, [1, 1500, 384]);  addmm_1 = None
        view_8: "bf16[1, 1500, 6, 64][576000, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_7, [1, -1, 6, 64]);  view_7 = None
        permute_6: "bf16[1, 6, 1500, 64][576000, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        clone_4: "bf16[1, 6, 1500, 64][576000, 96000, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format);  permute_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention = torch.ops.aten._scaled_dot_product_cudnn_attention.default(clone_2, clone_3, clone_4, None, False, scale = 1.0);  clone_2 = clone_3 = clone_4 = None
        getitem_2: "bf16[1, 6, 1500, 64][576000, 96000, 64, 1]cuda:0" = _scaled_dot_product_cudnn_attention[0];  _scaled_dot_product_cudnn_attention = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "bf16[1, 1500, 6, 64][576000, 64, 96000, 1]cuda:0" = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3]);  getitem_2 = None
        clone_5: "bf16[1, 1500, 6, 64][576000, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:353 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_9: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_5, [1, 1500, -1]);  clone_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        view_10: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_9, [1500, 384]);  view_9 = None
        permute_8: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_2: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg14_1, view_10, permute_8);  arg14_1 = view_10 = permute_8 = None
        view_11: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [1, 1500, 384]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:399 in forward, code: hidden_states = residual + hidden_states
        add_5: "bf16[1, 1500, 384][576000, 1, 1500]cuda:0" = torch.ops.aten.add.Tensor(add_2, view_11);  add_2 = view_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:402 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        clone_7: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.clone.default(add_5, memory_format = torch.contiguous_format)
        convert_element_type_17: "f32[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_7, torch.float32);  clone_7 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_17, [2], correction = 0, keepdim = True)
        getitem_11: "f32[1, 1500, 1][1500, 1, 1]cuda:0" = var_mean_1[0]
        getitem_12: "f32[1, 1500, 1][1500, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        sub_1: "f32[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_17, getitem_12);  convert_element_type_17 = getitem_12 = None
        add_6: "f32[1, 1500, 1][1500, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_11, 1e-05);  getitem_11 = None
        rsqrt_1: "f32[1, 1500, 1][1500, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        mul_9: "f32[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        mul_10: "f32[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_9, arg15_1);  mul_9 = arg15_1 = None
        add_7: "f32[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_10, arg16_1);  mul_10 = arg16_1 = None
        convert_element_type_18: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_7, torch.bfloat16);  add_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:403 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_12: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_18, [1500, 384]);  convert_element_type_18 = None
        permute_9: "bf16[384, 1536][1, 384]cuda:0" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        addmm_3: "bf16[1500, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg18_1, view_12, permute_9);  arg18_1 = view_12 = permute_9 = None
        view_13: "bf16[1, 1500, 1536][2304000, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [1, 1500, 1536]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_22: "f32[1, 1500, 1536][2304000, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_13, torch.float32);  view_13 = None
        mul_11: "f32[1, 1500, 1536][2304000, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_22, 0.5)
        mul_12: "f32[1, 1500, 1536][2304000, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_22, 0.7071067811865476);  convert_element_type_22 = None
        erf_2: "f32[1, 1500, 1536][2304000, 1536, 1]cuda:0" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_8: "f32[1, 1500, 1536][2304000, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_13: "f32[1, 1500, 1536][2304000, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_11, add_8);  mul_11 = add_8 = None
        convert_element_type_23: "bf16[1, 1500, 1536][2304000, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_13, torch.bfloat16);  mul_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:405 in forward, code: hidden_states = self.fc2(hidden_states)
        view_14: "bf16[1500, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_23, [1500, 1536]);  convert_element_type_23 = None
        permute_10: "bf16[1536, 384][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        addmm_4: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg20_1, view_14, permute_10);  arg20_1 = view_14 = permute_10 = None
        view_15: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [1, 1500, 384]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:407 in forward, code: hidden_states = residual + hidden_states
        add_9: "bf16[1, 1500, 384][576000, 1, 1500]cuda:0" = torch.ops.aten.add.Tensor(add_5, view_15);  add_5 = view_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:392 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        clone_10: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.clone.default(add_9, memory_format = torch.contiguous_format)
        convert_element_type_27: "f32[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_10, torch.float32);  clone_10 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_27, [2], correction = 0, keepdim = True)
        getitem_13: "f32[1, 1500, 1][1500, 1, 1]cuda:0" = var_mean_2[0]
        getitem_14: "f32[1, 1500, 1][1500, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None
        sub_2: "f32[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_27, getitem_14);  convert_element_type_27 = getitem_14 = None
        add_10: "f32[1, 1500, 1][1500, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_13, 1e-05);  getitem_13 = None
        rsqrt_2: "f32[1, 1500, 1][1500, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        mul_14: "f32[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
        mul_15: "f32[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, arg21_1);  mul_14 = arg21_1 = None
        add_11: "f32[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_15, arg22_1);  mul_15 = arg22_1 = None
        convert_element_type_28: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_11, torch.bfloat16);  add_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        view_16: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_28, [1500, 384])
        permute_11: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        addmm_5: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg24_1, view_16, permute_11);  arg24_1 = view_16 = permute_11 = None
        view_17: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_5, [1, 1500, 384]);  addmm_5 = None
        mul_16: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_17, 0.125);  view_17 = None
        view_18: "bf16[1, 1500, 6, 64][576000, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_16, [1, 1500, -1, 64]);  mul_16 = None
        permute_12: "bf16[1, 6, 1500, 64][576000, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None
        clone_11: "bf16[1, 6, 1500, 64][576000, 96000, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_12, memory_format = torch.contiguous_format);  permute_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_19: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_28, [1500, 384])
        permute_13: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        mm_1: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_19, permute_13);  view_19 = permute_13 = None
        view_20: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_1, [1, 1500, 384]);  mm_1 = None
        view_21: "bf16[1, 1500, 6, 64][576000, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_20, [1, -1, 6, 64]);  view_20 = None
        permute_14: "bf16[1, 6, 1500, 64][576000, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_21, [0, 2, 1, 3]);  view_21 = None
        clone_12: "bf16[1, 6, 1500, 64][576000, 96000, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_22: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_28, [1500, 384]);  convert_element_type_28 = None
        permute_15: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        addmm_6: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg27_1, view_22, permute_15);  arg27_1 = view_22 = permute_15 = None
        view_23: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [1, 1500, 384]);  addmm_6 = None
        view_24: "bf16[1, 1500, 6, 64][576000, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_23, [1, -1, 6, 64]);  view_23 = None
        permute_16: "bf16[1, 6, 1500, 64][576000, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
        clone_13: "bf16[1, 6, 1500, 64][576000, 96000, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_1 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(clone_11, clone_12, clone_13, None, False, scale = 1.0);  clone_11 = clone_12 = clone_13 = None
        getitem_15: "bf16[1, 6, 1500, 64][576000, 96000, 64, 1]cuda:0" = _scaled_dot_product_cudnn_attention_1[0];  _scaled_dot_product_cudnn_attention_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_17: "bf16[1, 1500, 6, 64][576000, 64, 96000, 1]cuda:0" = torch.ops.aten.permute.default(getitem_15, [0, 2, 1, 3]);  getitem_15 = None
        clone_14: "bf16[1, 1500, 6, 64][576000, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_17, memory_format = torch.contiguous_format);  permute_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:353 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_25: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_14, [1, 1500, -1]);  clone_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        view_26: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_25, [1500, 384]);  view_25 = None
        permute_18: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        addmm_7: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg29_1, view_26, permute_18);  arg29_1 = view_26 = permute_18 = None
        view_27: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_7, [1, 1500, 384]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:399 in forward, code: hidden_states = residual + hidden_states
        add_12: "bf16[1, 1500, 384][576000, 1, 1500]cuda:0" = torch.ops.aten.add.Tensor(add_9, view_27);  add_9 = view_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:402 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        clone_16: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.clone.default(add_12, memory_format = torch.contiguous_format)
        convert_element_type_40: "f32[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_16, torch.float32);  clone_16 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_40, [2], correction = 0, keepdim = True)
        getitem_24: "f32[1, 1500, 1][1500, 1, 1]cuda:0" = var_mean_3[0]
        getitem_25: "f32[1, 1500, 1][1500, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        sub_3: "f32[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_40, getitem_25);  convert_element_type_40 = getitem_25 = None
        add_13: "f32[1, 1500, 1][1500, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_3: "f32[1, 1500, 1][1500, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        mul_17: "f32[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = rsqrt_3 = None
        mul_18: "f32[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_17, arg30_1);  mul_17 = arg30_1 = None
        add_14: "f32[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_18, arg31_1);  mul_18 = arg31_1 = None
        convert_element_type_41: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_14, torch.bfloat16);  add_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:403 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_28: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_41, [1500, 384]);  convert_element_type_41 = None
        permute_19: "bf16[384, 1536][1, 384]cuda:0" = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
        addmm_8: "bf16[1500, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg33_1, view_28, permute_19);  arg33_1 = view_28 = permute_19 = None
        view_29: "bf16[1, 1500, 1536][2304000, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_8, [1, 1500, 1536]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_45: "f32[1, 1500, 1536][2304000, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_29, torch.float32);  view_29 = None
        mul_19: "f32[1, 1500, 1536][2304000, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_45, 0.5)
        mul_20: "f32[1, 1500, 1536][2304000, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_45, 0.7071067811865476);  convert_element_type_45 = None
        erf_3: "f32[1, 1500, 1536][2304000, 1536, 1]cuda:0" = torch.ops.aten.erf.default(mul_20);  mul_20 = None
        add_15: "f32[1, 1500, 1536][2304000, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_21: "f32[1, 1500, 1536][2304000, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_19, add_15);  mul_19 = add_15 = None
        convert_element_type_46: "bf16[1, 1500, 1536][2304000, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_21, torch.bfloat16);  mul_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:405 in forward, code: hidden_states = self.fc2(hidden_states)
        view_30: "bf16[1500, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_46, [1500, 1536]);  convert_element_type_46 = None
        permute_20: "bf16[1536, 384][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        addmm_9: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg35_1, view_30, permute_20);  arg35_1 = view_30 = permute_20 = None
        view_31: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_9, [1, 1500, 384]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:407 in forward, code: hidden_states = residual + hidden_states
        add_16: "bf16[1, 1500, 384][576000, 1, 1500]cuda:0" = torch.ops.aten.add.Tensor(add_12, view_31);  add_12 = view_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:392 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        clone_19: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.clone.default(add_16, memory_format = torch.contiguous_format)
        convert_element_type_50: "f32[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_19, torch.float32);  clone_19 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_50, [2], correction = 0, keepdim = True)
        getitem_26: "f32[1, 1500, 1][1500, 1, 1]cuda:0" = var_mean_4[0]
        getitem_27: "f32[1, 1500, 1][1500, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None
        sub_4: "f32[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_50, getitem_27);  convert_element_type_50 = getitem_27 = None
        add_17: "f32[1, 1500, 1][1500, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_4: "f32[1, 1500, 1][1500, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        mul_22: "f32[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = rsqrt_4 = None
        mul_23: "f32[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_22, arg36_1);  mul_22 = arg36_1 = None
        add_18: "f32[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_23, arg37_1);  mul_23 = arg37_1 = None
        convert_element_type_51: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_18, torch.bfloat16);  add_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        view_32: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_51, [1500, 384])
        permute_21: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        addmm_10: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg39_1, view_32, permute_21);  arg39_1 = view_32 = permute_21 = None
        view_33: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [1, 1500, 384]);  addmm_10 = None
        mul_24: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_33, 0.125);  view_33 = None
        view_34: "bf16[1, 1500, 6, 64][576000, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_24, [1, 1500, -1, 64]);  mul_24 = None
        permute_22: "bf16[1, 6, 1500, 64][576000, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None
        clone_20: "bf16[1, 6, 1500, 64][576000, 96000, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_22, memory_format = torch.contiguous_format);  permute_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_35: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_51, [1500, 384])
        permute_23: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
        mm_2: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_35, permute_23);  view_35 = permute_23 = None
        view_36: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_2, [1, 1500, 384]);  mm_2 = None
        view_37: "bf16[1, 1500, 6, 64][576000, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_36, [1, -1, 6, 64]);  view_36 = None
        permute_24: "bf16[1, 6, 1500, 64][576000, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_37, [0, 2, 1, 3]);  view_37 = None
        clone_21: "bf16[1, 6, 1500, 64][576000, 96000, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_38: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_51, [1500, 384]);  convert_element_type_51 = None
        permute_25: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        addmm_11: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg42_1, view_38, permute_25);  arg42_1 = view_38 = permute_25 = None
        view_39: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_11, [1, 1500, 384]);  addmm_11 = None
        view_40: "bf16[1, 1500, 6, 64][576000, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_39, [1, -1, 6, 64]);  view_39 = None
        permute_26: "bf16[1, 6, 1500, 64][576000, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
        clone_22: "bf16[1, 6, 1500, 64][576000, 96000, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_2 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(clone_20, clone_21, clone_22, None, False, scale = 1.0);  clone_20 = clone_21 = clone_22 = None
        getitem_28: "bf16[1, 6, 1500, 64][576000, 96000, 64, 1]cuda:0" = _scaled_dot_product_cudnn_attention_2[0];  _scaled_dot_product_cudnn_attention_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_27: "bf16[1, 1500, 6, 64][576000, 64, 96000, 1]cuda:0" = torch.ops.aten.permute.default(getitem_28, [0, 2, 1, 3]);  getitem_28 = None
        clone_23: "bf16[1, 1500, 6, 64][576000, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:353 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_41: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_23, [1, 1500, -1]);  clone_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        view_42: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_41, [1500, 384]);  view_41 = None
        permute_28: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_12: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg44_1, view_42, permute_28);  arg44_1 = view_42 = permute_28 = None
        view_43: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_12, [1, 1500, 384]);  addmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:399 in forward, code: hidden_states = residual + hidden_states
        add_19: "bf16[1, 1500, 384][576000, 1, 1500]cuda:0" = torch.ops.aten.add.Tensor(add_16, view_43);  add_16 = view_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:402 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        clone_25: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.clone.default(add_19, memory_format = torch.contiguous_format)
        convert_element_type_63: "f32[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_25, torch.float32);  clone_25 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_63, [2], correction = 0, keepdim = True)
        getitem_37: "f32[1, 1500, 1][1500, 1, 1]cuda:0" = var_mean_5[0]
        getitem_38: "f32[1, 1500, 1][1500, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        sub_5: "f32[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_63, getitem_38);  convert_element_type_63 = getitem_38 = None
        add_20: "f32[1, 1500, 1][1500, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_37, 1e-05);  getitem_37 = None
        rsqrt_5: "f32[1, 1500, 1][1500, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        mul_25: "f32[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = rsqrt_5 = None
        mul_26: "f32[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_25, arg45_1);  mul_25 = arg45_1 = None
        add_21: "f32[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_26, arg46_1);  mul_26 = arg46_1 = None
        convert_element_type_64: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_21, torch.bfloat16);  add_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:403 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_44: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_64, [1500, 384]);  convert_element_type_64 = None
        permute_29: "bf16[384, 1536][1, 384]cuda:0" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_13: "bf16[1500, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg48_1, view_44, permute_29);  arg48_1 = view_44 = permute_29 = None
        view_45: "bf16[1, 1500, 1536][2304000, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_13, [1, 1500, 1536]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_68: "f32[1, 1500, 1536][2304000, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_45, torch.float32);  view_45 = None
        mul_27: "f32[1, 1500, 1536][2304000, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_68, 0.5)
        mul_28: "f32[1, 1500, 1536][2304000, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_68, 0.7071067811865476);  convert_element_type_68 = None
        erf_4: "f32[1, 1500, 1536][2304000, 1536, 1]cuda:0" = torch.ops.aten.erf.default(mul_28);  mul_28 = None
        add_22: "f32[1, 1500, 1536][2304000, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_29: "f32[1, 1500, 1536][2304000, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_27, add_22);  mul_27 = add_22 = None
        convert_element_type_69: "bf16[1, 1500, 1536][2304000, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_29, torch.bfloat16);  mul_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:405 in forward, code: hidden_states = self.fc2(hidden_states)
        view_46: "bf16[1500, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_69, [1500, 1536]);  convert_element_type_69 = None
        permute_30: "bf16[1536, 384][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        addmm_14: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg50_1, view_46, permute_30);  arg50_1 = view_46 = permute_30 = None
        view_47: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_14, [1, 1500, 384]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:407 in forward, code: hidden_states = residual + hidden_states
        add_23: "bf16[1, 1500, 384][576000, 1, 1500]cuda:0" = torch.ops.aten.add.Tensor(add_19, view_47);  add_19 = view_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:392 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        clone_28: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.clone.default(add_23, memory_format = torch.contiguous_format)
        convert_element_type_73: "f32[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_28, torch.float32);  clone_28 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_73, [2], correction = 0, keepdim = True)
        getitem_39: "f32[1, 1500, 1][1500, 1, 1]cuda:0" = var_mean_6[0]
        getitem_40: "f32[1, 1500, 1][1500, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None
        sub_6: "f32[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_73, getitem_40);  convert_element_type_73 = getitem_40 = None
        add_24: "f32[1, 1500, 1][1500, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_39, 1e-05);  getitem_39 = None
        rsqrt_6: "f32[1, 1500, 1][1500, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        mul_30: "f32[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = rsqrt_6 = None
        mul_31: "f32[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_30, arg51_1);  mul_30 = arg51_1 = None
        add_25: "f32[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_31, arg52_1);  mul_31 = arg52_1 = None
        convert_element_type_74: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_25, torch.bfloat16);  add_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        view_48: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_74, [1500, 384])
        permute_31: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        addmm_15: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg54_1, view_48, permute_31);  arg54_1 = view_48 = permute_31 = None
        view_49: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_15, [1, 1500, 384]);  addmm_15 = None
        mul_32: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_49, 0.125);  view_49 = None
        view_50: "bf16[1, 1500, 6, 64][576000, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_32, [1, 1500, -1, 64]);  mul_32 = None
        permute_32: "bf16[1, 6, 1500, 64][576000, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
        clone_29: "bf16[1, 6, 1500, 64][576000, 96000, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_32, memory_format = torch.contiguous_format);  permute_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_51: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_74, [1500, 384])
        permute_33: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        mm_3: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_51, permute_33);  view_51 = permute_33 = None
        view_52: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_3, [1, 1500, 384]);  mm_3 = None
        view_53: "bf16[1, 1500, 6, 64][576000, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_52, [1, -1, 6, 64]);  view_52 = None
        permute_34: "bf16[1, 6, 1500, 64][576000, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
        clone_30: "bf16[1, 6, 1500, 64][576000, 96000, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format);  permute_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_54: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_74, [1500, 384]);  convert_element_type_74 = None
        permute_35: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        addmm_16: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg57_1, view_54, permute_35);  arg57_1 = view_54 = permute_35 = None
        view_55: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [1, 1500, 384]);  addmm_16 = None
        view_56: "bf16[1, 1500, 6, 64][576000, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_55, [1, -1, 6, 64]);  view_55 = None
        permute_36: "bf16[1, 6, 1500, 64][576000, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_56, [0, 2, 1, 3]);  view_56 = None
        clone_31: "bf16[1, 6, 1500, 64][576000, 96000, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_36, memory_format = torch.contiguous_format);  permute_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_3 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(clone_29, clone_30, clone_31, None, False, scale = 1.0);  clone_29 = clone_30 = clone_31 = None
        getitem_41: "bf16[1, 6, 1500, 64][576000, 96000, 64, 1]cuda:0" = _scaled_dot_product_cudnn_attention_3[0];  _scaled_dot_product_cudnn_attention_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_37: "bf16[1, 1500, 6, 64][576000, 64, 96000, 1]cuda:0" = torch.ops.aten.permute.default(getitem_41, [0, 2, 1, 3]);  getitem_41 = None
        clone_32: "bf16[1, 1500, 6, 64][576000, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:353 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_57: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_32, [1, 1500, -1]);  clone_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        view_58: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_57, [1500, 384]);  view_57 = None
        permute_38: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        addmm_17: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg59_1, view_58, permute_38);  arg59_1 = view_58 = permute_38 = None
        view_59: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_17, [1, 1500, 384]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:399 in forward, code: hidden_states = residual + hidden_states
        add_26: "bf16[1, 1500, 384][576000, 1, 1500]cuda:0" = torch.ops.aten.add.Tensor(add_23, view_59);  add_23 = view_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:402 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        clone_34: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.clone.default(add_26, memory_format = torch.contiguous_format)
        convert_element_type_86: "f32[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_34, torch.float32);  clone_34 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_86, [2], correction = 0, keepdim = True)
        getitem_50: "f32[1, 1500, 1][1500, 1, 1]cuda:0" = var_mean_7[0]
        getitem_51: "f32[1, 1500, 1][1500, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None
        sub_7: "f32[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_86, getitem_51);  convert_element_type_86 = getitem_51 = None
        add_27: "f32[1, 1500, 1][1500, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
        rsqrt_7: "f32[1, 1500, 1][1500, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        mul_33: "f32[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = rsqrt_7 = None
        mul_34: "f32[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_33, arg60_1);  mul_33 = arg60_1 = None
        add_28: "f32[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_34, arg61_1);  mul_34 = arg61_1 = None
        convert_element_type_87: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_28, torch.bfloat16);  add_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:403 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_60: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_87, [1500, 384]);  convert_element_type_87 = None
        permute_39: "bf16[384, 1536][1, 384]cuda:0" = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
        addmm_18: "bf16[1500, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg63_1, view_60, permute_39);  arg63_1 = view_60 = permute_39 = None
        view_61: "bf16[1, 1500, 1536][2304000, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_18, [1, 1500, 1536]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_91: "f32[1, 1500, 1536][2304000, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_61, torch.float32);  view_61 = None
        mul_35: "f32[1, 1500, 1536][2304000, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_91, 0.5)
        mul_36: "f32[1, 1500, 1536][2304000, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_91, 0.7071067811865476);  convert_element_type_91 = None
        erf_5: "f32[1, 1500, 1536][2304000, 1536, 1]cuda:0" = torch.ops.aten.erf.default(mul_36);  mul_36 = None
        add_29: "f32[1, 1500, 1536][2304000, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_37: "f32[1, 1500, 1536][2304000, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_35, add_29);  mul_35 = add_29 = None
        convert_element_type_92: "bf16[1, 1500, 1536][2304000, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_37, torch.bfloat16);  mul_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:405 in forward, code: hidden_states = self.fc2(hidden_states)
        view_62: "bf16[1500, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_92, [1500, 1536]);  convert_element_type_92 = None
        permute_40: "bf16[1536, 384][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        addmm_19: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg65_1, view_62, permute_40);  arg65_1 = view_62 = permute_40 = None
        view_63: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_19, [1, 1500, 384]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:407 in forward, code: hidden_states = residual + hidden_states
        add_30: "bf16[1, 1500, 384][576000, 1, 1500]cuda:0" = torch.ops.aten.add.Tensor(add_26, view_63);  add_26 = view_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:642 in forward, code: hidden_states = self.layer_norm(hidden_states)
        clone_37: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.clone.default(add_30, memory_format = torch.contiguous_format);  add_30 = None
        convert_element_type_96: "f32[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_37, torch.float32);  clone_37 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_96, [2], correction = 0, keepdim = True)
        getitem_52: "f32[1, 1500, 1][1500, 1, 1]cuda:0" = var_mean_8[0]
        getitem_53: "f32[1, 1500, 1][1500, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:749 in forward, code: position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_key_values_length
        iota_1: "i64[1][1]cuda:0" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_33: "i64[1][1]cuda:0" = torch.ops.aten.add.Tensor(iota_1, 0);  iota_1 = add_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:737 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        embedding_1: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.embedding.default(arg69_1, arg68_1, 50257);  arg68_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:750 in forward, code: position_ids = position_ids.unsqueeze(0).repeat(inputs_embeds.shape[0], 1)
        full_default: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.full.default([1, 1], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:212 in forward, code: return self.weight[position_ids]
        index: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.index.Tensor(arg70_1, [full_default]);  arg70_1 = full_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:762 in forward, code: hidden_states = inputs_embeds + positions.to(inputs_embeds.device)
        add_34: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(embedding_1, index);  embedding_1 = index = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:470 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_98: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_34, torch.float32)
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_98, [2], correction = 0, keepdim = True)
        getitem_54: "f32[1, 1, 1][1, 1, 1]cuda:0" = var_mean_9[0]
        getitem_55: "f32[1, 1, 1][1, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None
        sub_9: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_98, getitem_55);  convert_element_type_98 = getitem_55 = None
        add_37: "f32[1, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
        rsqrt_9: "f32[1, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        mul_40: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = rsqrt_9 = None
        mul_41: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_40, arg71_1);  mul_40 = arg71_1 = None
        add_38: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_41, arg72_1);  mul_41 = arg72_1 = None
        convert_element_type_99: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_38, torch.bfloat16);  add_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        view_64: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_99, [1, 384])
        permute_41: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_20: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg74_1, view_64, permute_41);  arg74_1 = view_64 = permute_41 = None
        view_65: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_20, [1, 1, 384]);  addmm_20 = None
        mul_42: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_65, 0.125);  view_65 = None
        view_66: "bf16[1, 1, 6, 64][384, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_42, [1, 1, -1, 64]);  mul_42 = None
        permute_42: "bf16[1, 6, 1, 64][384, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_67: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_99, [1, 384])
        permute_43: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        mm_4: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_67, permute_43);  view_67 = permute_43 = None
        view_68: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_4, [1, 1, 384]);  mm_4 = None
        view_69: "bf16[1, 1, 6, 64][384, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_68, [1, -1, 6, 64]);  view_68 = None
        permute_44: "bf16[1, 6, 1, 64][384, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_69, [0, 2, 1, 3]);  view_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_70: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_99, [1, 384]);  convert_element_type_99 = None
        permute_45: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        addmm_21: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg77_1, view_70, permute_45);  arg77_1 = view_70 = permute_45 = None
        view_71: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_21, [1, 1, 384]);  addmm_21 = None
        view_72: "bf16[1, 1, 6, 64][384, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_71, [1, -1, 6, 64]);  view_71 = None
        permute_46: "bf16[1, 6, 1, 64][384, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:512 in sdpa_mask, code: kv_arange = torch.arange(kv_length, device=device) + kv_offset
        iota_5: "i64[1][1]cuda:0" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_36: "i64[1][1]cuda:0" = torch.ops.aten.add.Tensor(iota_5, 0);  iota_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_4: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_36, 0);  add_36 = None
        unsqueeze_5: "i64[1, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 1);  unsqueeze_4 = None
        unsqueeze_6: "i64[1, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 2);  unsqueeze_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_4: "i64[1][1]cuda:0" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_35: "i64[1][1]cuda:0" = torch.ops.aten.add.Tensor(iota_4, 0);  iota_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_1: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_35, 0);  add_35 = None
        unsqueeze_2: "i64[1, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 1);  unsqueeze_1 = None
        unsqueeze_3: "i64[1, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 3);  unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le: "b8[1, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.le.Tensor(unsqueeze_6, unsqueeze_3);  unsqueeze_6 = unsqueeze_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand: "b8[1, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.expand.default(le, [1, -1, 1, 1]);  le = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_1: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0))
        where: "bf16[1, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.where.self(expand, scalar_tensor_1, scalar_tensor);  scalar_tensor_1 = scalar_tensor = None
        constant_pad_nd: "bf16[1, 1, 1, 8][8, 8, 8, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(where, [0, 7], 0.0);  where = None
        slice_1: "bf16[1, 1, 1, 1][8, 8, 8, 1]cuda:0" = torch.ops.aten.slice.Tensor(constant_pad_nd, -1, 0, 1);  constant_pad_nd = None
        expand_1: "bf16[1, 6, 1, 1][8, 0, 8, 1]cuda:0" = torch.ops.aten.expand.default(slice_1, [1, 6, 1, 1]);  slice_1 = None
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_42, permute_44, permute_46, expand_1, False, scale = 1.0);  permute_42 = expand_1 = None
        getitem_56: "bf16[1, 6, 1, 64][384, 64, 384, 1]cuda:0" = _scaled_dot_product_efficient_attention[0];  _scaled_dot_product_efficient_attention = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_47: "bf16[1, 1, 6, 64][384, 384, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_56, [0, 2, 1, 3]);  getitem_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:353 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_73: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_47, [1, 1, -1]);  permute_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        view_74: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_73, [1, 384]);  view_73 = None
        permute_48: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
        addmm_22: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg79_1, view_74, permute_48);  arg79_1 = view_74 = permute_48 = None
        view_75: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [1, 1, 384]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:480 in forward, code: hidden_states = residual + hidden_states
        add_39: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_34, view_75);  add_34 = view_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:485 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_111: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_39, torch.float32)
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_111, [2], correction = 0, keepdim = True)
        getitem_60: "f32[1, 1, 1][1, 1, 1]cuda:0" = var_mean_10[0]
        getitem_61: "f32[1, 1, 1][1, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None
        sub_10: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_111, getitem_61);  convert_element_type_111 = getitem_61 = None
        add_40: "f32[1, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
        rsqrt_10: "f32[1, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        mul_43: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = rsqrt_10 = None
        mul_44: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_43, arg80_1);  mul_43 = arg80_1 = None
        add_41: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_44, arg81_1);  mul_44 = arg81_1 = None
        convert_element_type_112: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_41, torch.bfloat16);  add_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        view_76: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_112, [1, 384]);  convert_element_type_112 = None
        permute_49: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        addmm_23: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg83_1, view_76, permute_49);  arg83_1 = view_76 = permute_49 = None
        view_77: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_23, [1, 1, 384]);  addmm_23 = None
        mul_45: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_77, 0.125);  view_77 = None
        view_78: "bf16[1, 1, 6, 64][384, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_45, [1, 1, -1, 64]);  mul_45 = None
        permute_50: "bf16[1, 6, 1, 64][384, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:642 in forward, code: hidden_states = self.layer_norm(hidden_states)
        sub_8: "f32[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_96, getitem_53);  convert_element_type_96 = getitem_53 = None
        add_31: "f32[1, 1500, 1][1500, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
        rsqrt_8: "f32[1, 1500, 1][1500, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        mul_38: "f32[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = rsqrt_8 = None
        mul_39: "f32[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_38, arg66_1);  mul_38 = arg66_1 = None
        add_32: "f32[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_39, arg67_1);  mul_39 = arg67_1 = None
        convert_element_type_97: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_32, torch.bfloat16);  add_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_79: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_97, [1500, 384])
        permute_51: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
        mm_5: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_79, permute_51);  view_79 = permute_51 = None
        view_80: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_5, [1, 1500, 384]);  mm_5 = None
        view_81: "bf16[1, 1500, 6, 64][576000, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_80, [1, -1, 6, 64]);  view_80 = None
        permute_52: "bf16[1, 6, 1500, 64][576000, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_81, [0, 2, 1, 3]);  view_81 = None
        clone_42: "bf16[1, 6, 1500, 64][576000, 96000, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_82: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_97, [1500, 384])
        permute_53: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        addmm_24: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg86_1, view_82, permute_53);  arg86_1 = view_82 = permute_53 = None
        view_83: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_24, [1, 1500, 384]);  addmm_24 = None
        view_84: "bf16[1, 1500, 6, 64][576000, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_83, [1, -1, 6, 64]);  view_83 = None
        permute_54: "bf16[1, 6, 1500, 64][576000, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_84, [0, 2, 1, 3]);  view_84 = None
        clone_43: "bf16[1, 6, 1500, 64][576000, 96000, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_54, memory_format = torch.contiguous_format);  permute_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_4 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_50, clone_42, clone_43, None, False, scale = 1.0);  permute_50 = None
        getitem_62: "bf16[1, 6, 1, 64][384, 64, 64, 1]cuda:0" = _scaled_dot_product_cudnn_attention_4[0];  _scaled_dot_product_cudnn_attention_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_55: "bf16[1, 1, 6, 64][384, 64, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_62, [0, 2, 1, 3]);  getitem_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:353 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_85: "bf16[1, 1, 384][384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_55, [1, 1, -1]);  permute_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        view_86: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_85, [1, 384]);  view_85 = None
        permute_56: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        addmm_25: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg88_1, view_86, permute_56);  arg88_1 = view_86 = permute_56 = None
        view_87: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_25, [1, 1, 384]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:494 in forward, code: hidden_states = residual + hidden_states
        add_42: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_39, view_87);  add_39 = view_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:498 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_124: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_42, torch.float32)
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_124, [2], correction = 0, keepdim = True)
        getitem_71: "f32[1, 1, 1][1, 1, 1]cuda:0" = var_mean_11[0]
        getitem_72: "f32[1, 1, 1][1, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None
        sub_11: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_124, getitem_72);  convert_element_type_124 = getitem_72 = None
        add_43: "f32[1, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_71, 1e-05);  getitem_71 = None
        rsqrt_11: "f32[1, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        mul_46: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = rsqrt_11 = None
        mul_47: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_46, arg89_1);  mul_46 = arg89_1 = None
        add_44: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_47, arg90_1);  mul_47 = arg90_1 = None
        convert_element_type_125: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_44, torch.bfloat16);  add_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:499 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_88: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_125, [1, 384]);  convert_element_type_125 = None
        permute_57: "bf16[384, 1536][1, 384]cuda:0" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_26: "bf16[1, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg92_1, view_88, permute_57);  arg92_1 = view_88 = permute_57 = None
        view_89: "bf16[1, 1, 1536][1536, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_26, [1, 1, 1536]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_129: "f32[1, 1, 1536][1536, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_89, torch.float32);  view_89 = None
        mul_48: "f32[1, 1, 1536][1536, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_129, 0.5)
        mul_49: "f32[1, 1, 1536][1536, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_129, 0.7071067811865476);  convert_element_type_129 = None
        erf_6: "f32[1, 1, 1536][1536, 1536, 1]cuda:0" = torch.ops.aten.erf.default(mul_49);  mul_49 = None
        add_45: "f32[1, 1, 1536][1536, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_50: "f32[1, 1, 1536][1536, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_48, add_45);  mul_48 = add_45 = None
        convert_element_type_130: "bf16[1, 1, 1536][1536, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_50, torch.bfloat16);  mul_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:501 in forward, code: hidden_states = self.fc2(hidden_states)
        view_90: "bf16[1, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_130, [1, 1536]);  convert_element_type_130 = None
        permute_58: "bf16[1536, 384][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        addmm_27: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg94_1, view_90, permute_58);  arg94_1 = view_90 = permute_58 = None
        view_91: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_27, [1, 1, 384]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:503 in forward, code: hidden_states = residual + hidden_states
        add_46: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_42, view_91);  add_42 = view_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:470 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_134: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_46, torch.float32)
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_134, [2], correction = 0, keepdim = True)
        getitem_73: "f32[1, 1, 1][1, 1, 1]cuda:0" = var_mean_12[0]
        getitem_74: "f32[1, 1, 1][1, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None
        sub_12: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_134, getitem_74);  convert_element_type_134 = getitem_74 = None
        add_47: "f32[1, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_73, 1e-05);  getitem_73 = None
        rsqrt_12: "f32[1, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
        mul_51: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = rsqrt_12 = None
        mul_52: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_51, arg95_1);  mul_51 = arg95_1 = None
        add_48: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_52, arg96_1);  mul_52 = arg96_1 = None
        convert_element_type_135: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_48, torch.bfloat16);  add_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        view_92: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_135, [1, 384])
        permute_59: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        addmm_28: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg98_1, view_92, permute_59);  arg98_1 = view_92 = permute_59 = None
        view_93: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_28, [1, 1, 384]);  addmm_28 = None
        mul_53: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_93, 0.125);  view_93 = None
        view_94: "bf16[1, 1, 6, 64][384, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_53, [1, 1, -1, 64]);  mul_53 = None
        permute_60: "bf16[1, 6, 1, 64][384, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_95: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_135, [1, 384])
        permute_61: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        mm_6: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_95, permute_61);  view_95 = permute_61 = None
        view_96: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_6, [1, 1, 384]);  mm_6 = None
        view_97: "bf16[1, 1, 6, 64][384, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_96, [1, -1, 6, 64]);  view_96 = None
        permute_62: "bf16[1, 6, 1, 64][384, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_97, [0, 2, 1, 3]);  view_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_98: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_135, [1, 384]);  convert_element_type_135 = None
        permute_63: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
        addmm_29: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg101_1, view_98, permute_63);  arg101_1 = view_98 = permute_63 = None
        view_99: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_29, [1, 1, 384]);  addmm_29 = None
        view_100: "bf16[1, 1, 6, 64][384, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_99, [1, -1, 6, 64]);  view_99 = None
        permute_64: "bf16[1, 6, 1, 64][384, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_100, [0, 2, 1, 3]);  view_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_3: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_2: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0))
        where_1: "bf16[1, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.where.self(expand, scalar_tensor_3, scalar_tensor_2);  scalar_tensor_3 = scalar_tensor_2 = None
        constant_pad_nd_1: "bf16[1, 1, 1, 8][8, 8, 8, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(where_1, [0, 7], 0.0);  where_1 = None
        slice_2: "bf16[1, 1, 1, 1][8, 8, 8, 1]cuda:0" = torch.ops.aten.slice.Tensor(constant_pad_nd_1, -1, 0, 1);  constant_pad_nd_1 = None
        expand_2: "bf16[1, 6, 1, 1][8, 0, 8, 1]cuda:0" = torch.ops.aten.expand.default(slice_2, [1, 6, 1, 1]);  slice_2 = None
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_60, permute_62, permute_64, expand_2, False, scale = 1.0);  permute_60 = expand_2 = None
        getitem_75: "bf16[1, 6, 1, 64][384, 64, 384, 1]cuda:0" = _scaled_dot_product_efficient_attention_1[0];  _scaled_dot_product_efficient_attention_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_65: "bf16[1, 1, 6, 64][384, 384, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_75, [0, 2, 1, 3]);  getitem_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:353 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_101: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_65, [1, 1, -1]);  permute_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        view_102: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_101, [1, 384]);  view_101 = None
        permute_66: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
        addmm_30: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg103_1, view_102, permute_66);  arg103_1 = view_102 = permute_66 = None
        view_103: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_30, [1, 1, 384]);  addmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:480 in forward, code: hidden_states = residual + hidden_states
        add_49: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_46, view_103);  add_46 = view_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:485 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_147: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_49, torch.float32)
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_147, [2], correction = 0, keepdim = True)
        getitem_79: "f32[1, 1, 1][1, 1, 1]cuda:0" = var_mean_13[0]
        getitem_80: "f32[1, 1, 1][1, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None
        sub_13: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_147, getitem_80);  convert_element_type_147 = getitem_80 = None
        add_50: "f32[1, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_79, 1e-05);  getitem_79 = None
        rsqrt_13: "f32[1, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        mul_54: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = rsqrt_13 = None
        mul_55: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_54, arg104_1);  mul_54 = arg104_1 = None
        add_51: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_55, arg105_1);  mul_55 = arg105_1 = None
        convert_element_type_148: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_51, torch.bfloat16);  add_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        view_104: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_148, [1, 384]);  convert_element_type_148 = None
        permute_67: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
        addmm_31: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg107_1, view_104, permute_67);  arg107_1 = view_104 = permute_67 = None
        view_105: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_31, [1, 1, 384]);  addmm_31 = None
        mul_56: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_105, 0.125);  view_105 = None
        view_106: "bf16[1, 1, 6, 64][384, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_56, [1, 1, -1, 64]);  mul_56 = None
        permute_68: "bf16[1, 6, 1, 64][384, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_106, [0, 2, 1, 3]);  view_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_107: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_97, [1500, 384])
        permute_69: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        mm_7: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_107, permute_69);  view_107 = permute_69 = None
        view_108: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_7, [1, 1500, 384]);  mm_7 = None
        view_109: "bf16[1, 1500, 6, 64][576000, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_108, [1, -1, 6, 64]);  view_108 = None
        permute_70: "bf16[1, 6, 1500, 64][576000, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_109, [0, 2, 1, 3]);  view_109 = None
        clone_52: "bf16[1, 6, 1500, 64][576000, 96000, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_110: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_97, [1500, 384])
        permute_71: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        addmm_32: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg110_1, view_110, permute_71);  arg110_1 = view_110 = permute_71 = None
        view_111: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_32, [1, 1500, 384]);  addmm_32 = None
        view_112: "bf16[1, 1500, 6, 64][576000, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_111, [1, -1, 6, 64]);  view_111 = None
        permute_72: "bf16[1, 6, 1500, 64][576000, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None
        clone_53: "bf16[1, 6, 1500, 64][576000, 96000, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_72, memory_format = torch.contiguous_format);  permute_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_5 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_68, clone_52, clone_53, None, False, scale = 1.0);  permute_68 = None
        getitem_81: "bf16[1, 6, 1, 64][384, 64, 64, 1]cuda:0" = _scaled_dot_product_cudnn_attention_5[0];  _scaled_dot_product_cudnn_attention_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_73: "bf16[1, 1, 6, 64][384, 64, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_81, [0, 2, 1, 3]);  getitem_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:353 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_113: "bf16[1, 1, 384][384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_73, [1, 1, -1]);  permute_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        view_114: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_113, [1, 384]);  view_113 = None
        permute_74: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
        addmm_33: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg112_1, view_114, permute_74);  arg112_1 = view_114 = permute_74 = None
        view_115: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_33, [1, 1, 384]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:494 in forward, code: hidden_states = residual + hidden_states
        add_52: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_49, view_115);  add_49 = view_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:498 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_160: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_52, torch.float32)
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_160, [2], correction = 0, keepdim = True)
        getitem_90: "f32[1, 1, 1][1, 1, 1]cuda:0" = var_mean_14[0]
        getitem_91: "f32[1, 1, 1][1, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None
        sub_14: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_160, getitem_91);  convert_element_type_160 = getitem_91 = None
        add_53: "f32[1, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
        rsqrt_14: "f32[1, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        mul_57: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = rsqrt_14 = None
        mul_58: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_57, arg113_1);  mul_57 = arg113_1 = None
        add_54: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_58, arg114_1);  mul_58 = arg114_1 = None
        convert_element_type_161: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_54, torch.bfloat16);  add_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:499 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_116: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_161, [1, 384]);  convert_element_type_161 = None
        permute_75: "bf16[384, 1536][1, 384]cuda:0" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        addmm_34: "bf16[1, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg116_1, view_116, permute_75);  arg116_1 = view_116 = permute_75 = None
        view_117: "bf16[1, 1, 1536][1536, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [1, 1, 1536]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_165: "f32[1, 1, 1536][1536, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_117, torch.float32);  view_117 = None
        mul_59: "f32[1, 1, 1536][1536, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_165, 0.5)
        mul_60: "f32[1, 1, 1536][1536, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_165, 0.7071067811865476);  convert_element_type_165 = None
        erf_7: "f32[1, 1, 1536][1536, 1536, 1]cuda:0" = torch.ops.aten.erf.default(mul_60);  mul_60 = None
        add_55: "f32[1, 1, 1536][1536, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_61: "f32[1, 1, 1536][1536, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_59, add_55);  mul_59 = add_55 = None
        convert_element_type_166: "bf16[1, 1, 1536][1536, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_61, torch.bfloat16);  mul_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:501 in forward, code: hidden_states = self.fc2(hidden_states)
        view_118: "bf16[1, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_166, [1, 1536]);  convert_element_type_166 = None
        permute_76: "bf16[1536, 384][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        addmm_35: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg118_1, view_118, permute_76);  arg118_1 = view_118 = permute_76 = None
        view_119: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_35, [1, 1, 384]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:503 in forward, code: hidden_states = residual + hidden_states
        add_56: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_52, view_119);  add_52 = view_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:470 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_170: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_56, torch.float32)
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_170, [2], correction = 0, keepdim = True)
        getitem_92: "f32[1, 1, 1][1, 1, 1]cuda:0" = var_mean_15[0]
        getitem_93: "f32[1, 1, 1][1, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None
        sub_15: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_170, getitem_93);  convert_element_type_170 = getitem_93 = None
        add_57: "f32[1, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_92, 1e-05);  getitem_92 = None
        rsqrt_15: "f32[1, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
        mul_62: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = rsqrt_15 = None
        mul_63: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_62, arg119_1);  mul_62 = arg119_1 = None
        add_58: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_63, arg120_1);  mul_63 = arg120_1 = None
        convert_element_type_171: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_58, torch.bfloat16);  add_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        view_120: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_171, [1, 384])
        permute_77: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_36: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg122_1, view_120, permute_77);  arg122_1 = view_120 = permute_77 = None
        view_121: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_36, [1, 1, 384]);  addmm_36 = None
        mul_64: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_121, 0.125);  view_121 = None
        view_122: "bf16[1, 1, 6, 64][384, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_64, [1, 1, -1, 64]);  mul_64 = None
        permute_78: "bf16[1, 6, 1, 64][384, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_122, [0, 2, 1, 3]);  view_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_123: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_171, [1, 384])
        permute_79: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        mm_8: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_123, permute_79);  view_123 = permute_79 = None
        view_124: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_8, [1, 1, 384]);  mm_8 = None
        view_125: "bf16[1, 1, 6, 64][384, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_124, [1, -1, 6, 64]);  view_124 = None
        permute_80: "bf16[1, 6, 1, 64][384, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_125, [0, 2, 1, 3]);  view_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_126: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_171, [1, 384]);  convert_element_type_171 = None
        permute_81: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
        addmm_37: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg125_1, view_126, permute_81);  arg125_1 = view_126 = permute_81 = None
        view_127: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_37, [1, 1, 384]);  addmm_37 = None
        view_128: "bf16[1, 1, 6, 64][384, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_127, [1, -1, 6, 64]);  view_127 = None
        permute_82: "bf16[1, 6, 1, 64][384, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_128, [0, 2, 1, 3]);  view_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_5: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_4: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0))
        where_2: "bf16[1, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.where.self(expand, scalar_tensor_5, scalar_tensor_4);  scalar_tensor_5 = scalar_tensor_4 = None
        constant_pad_nd_2: "bf16[1, 1, 1, 8][8, 8, 8, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(where_2, [0, 7], 0.0);  where_2 = None
        slice_3: "bf16[1, 1, 1, 1][8, 8, 8, 1]cuda:0" = torch.ops.aten.slice.Tensor(constant_pad_nd_2, -1, 0, 1);  constant_pad_nd_2 = None
        expand_3: "bf16[1, 6, 1, 1][8, 0, 8, 1]cuda:0" = torch.ops.aten.expand.default(slice_3, [1, 6, 1, 1]);  slice_3 = None
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_78, permute_80, permute_82, expand_3, False, scale = 1.0);  permute_78 = expand_3 = None
        getitem_94: "bf16[1, 6, 1, 64][384, 64, 384, 1]cuda:0" = _scaled_dot_product_efficient_attention_2[0];  _scaled_dot_product_efficient_attention_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_83: "bf16[1, 1, 6, 64][384, 384, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_94, [0, 2, 1, 3]);  getitem_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:353 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_129: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_83, [1, 1, -1]);  permute_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        view_130: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_129, [1, 384]);  view_129 = None
        permute_84: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
        addmm_38: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg127_1, view_130, permute_84);  arg127_1 = view_130 = permute_84 = None
        view_131: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_38, [1, 1, 384]);  addmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:480 in forward, code: hidden_states = residual + hidden_states
        add_59: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_56, view_131);  add_56 = view_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:485 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_183: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_59, torch.float32)
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_183, [2], correction = 0, keepdim = True)
        getitem_98: "f32[1, 1, 1][1, 1, 1]cuda:0" = var_mean_16[0]
        getitem_99: "f32[1, 1, 1][1, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None
        sub_16: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_183, getitem_99);  convert_element_type_183 = getitem_99 = None
        add_60: "f32[1, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_98, 1e-05);  getitem_98 = None
        rsqrt_16: "f32[1, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        mul_65: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = rsqrt_16 = None
        mul_66: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_65, arg128_1);  mul_65 = arg128_1 = None
        add_61: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_66, arg129_1);  mul_66 = arg129_1 = None
        convert_element_type_184: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_61, torch.bfloat16);  add_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        view_132: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_184, [1, 384]);  convert_element_type_184 = None
        permute_85: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
        addmm_39: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg131_1, view_132, permute_85);  arg131_1 = view_132 = permute_85 = None
        view_133: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_39, [1, 1, 384]);  addmm_39 = None
        mul_67: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_133, 0.125);  view_133 = None
        view_134: "bf16[1, 1, 6, 64][384, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_67, [1, 1, -1, 64]);  mul_67 = None
        permute_86: "bf16[1, 6, 1, 64][384, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_135: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_97, [1500, 384])
        permute_87: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
        mm_9: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_135, permute_87);  view_135 = permute_87 = None
        view_136: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_9, [1, 1500, 384]);  mm_9 = None
        view_137: "bf16[1, 1500, 6, 64][576000, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_136, [1, -1, 6, 64]);  view_136 = None
        permute_88: "bf16[1, 6, 1500, 64][576000, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None
        clone_62: "bf16[1, 6, 1500, 64][576000, 96000, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_88, memory_format = torch.contiguous_format);  permute_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_138: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_97, [1500, 384])
        permute_89: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        addmm_40: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg134_1, view_138, permute_89);  arg134_1 = view_138 = permute_89 = None
        view_139: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_40, [1, 1500, 384]);  addmm_40 = None
        view_140: "bf16[1, 1500, 6, 64][576000, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_139, [1, -1, 6, 64]);  view_139 = None
        permute_90: "bf16[1, 6, 1500, 64][576000, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None
        clone_63: "bf16[1, 6, 1500, 64][576000, 96000, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_6 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_86, clone_62, clone_63, None, False, scale = 1.0);  permute_86 = None
        getitem_100: "bf16[1, 6, 1, 64][384, 64, 64, 1]cuda:0" = _scaled_dot_product_cudnn_attention_6[0];  _scaled_dot_product_cudnn_attention_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_91: "bf16[1, 1, 6, 64][384, 64, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_100, [0, 2, 1, 3]);  getitem_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:353 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_141: "bf16[1, 1, 384][384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_91, [1, 1, -1]);  permute_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        view_142: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_141, [1, 384]);  view_141 = None
        permute_92: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        addmm_41: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg136_1, view_142, permute_92);  arg136_1 = view_142 = permute_92 = None
        view_143: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_41, [1, 1, 384]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:494 in forward, code: hidden_states = residual + hidden_states
        add_62: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_59, view_143);  add_59 = view_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:498 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_196: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_62, torch.float32)
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_196, [2], correction = 0, keepdim = True)
        getitem_109: "f32[1, 1, 1][1, 1, 1]cuda:0" = var_mean_17[0]
        getitem_110: "f32[1, 1, 1][1, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None
        sub_17: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_196, getitem_110);  convert_element_type_196 = getitem_110 = None
        add_63: "f32[1, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_109, 1e-05);  getitem_109 = None
        rsqrt_17: "f32[1, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
        mul_68: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = rsqrt_17 = None
        mul_69: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_68, arg137_1);  mul_68 = arg137_1 = None
        add_64: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_69, arg138_1);  mul_69 = arg138_1 = None
        convert_element_type_197: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_64, torch.bfloat16);  add_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:499 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_144: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_197, [1, 384]);  convert_element_type_197 = None
        permute_93: "bf16[384, 1536][1, 384]cuda:0" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_42: "bf16[1, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg140_1, view_144, permute_93);  arg140_1 = view_144 = permute_93 = None
        view_145: "bf16[1, 1, 1536][1536, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_42, [1, 1, 1536]);  addmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_201: "f32[1, 1, 1536][1536, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_145, torch.float32);  view_145 = None
        mul_70: "f32[1, 1, 1536][1536, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_201, 0.5)
        mul_71: "f32[1, 1, 1536][1536, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_201, 0.7071067811865476);  convert_element_type_201 = None
        erf_8: "f32[1, 1, 1536][1536, 1536, 1]cuda:0" = torch.ops.aten.erf.default(mul_71);  mul_71 = None
        add_65: "f32[1, 1, 1536][1536, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_72: "f32[1, 1, 1536][1536, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_70, add_65);  mul_70 = add_65 = None
        convert_element_type_202: "bf16[1, 1, 1536][1536, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_72, torch.bfloat16);  mul_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:501 in forward, code: hidden_states = self.fc2(hidden_states)
        view_146: "bf16[1, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_202, [1, 1536]);  convert_element_type_202 = None
        permute_94: "bf16[1536, 384][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        addmm_43: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg142_1, view_146, permute_94);  arg142_1 = view_146 = permute_94 = None
        view_147: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_43, [1, 1, 384]);  addmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:503 in forward, code: hidden_states = residual + hidden_states
        add_66: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_62, view_147);  add_62 = view_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:470 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type_206: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_66, torch.float32)
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_206, [2], correction = 0, keepdim = True)
        getitem_111: "f32[1, 1, 1][1, 1, 1]cuda:0" = var_mean_18[0]
        getitem_112: "f32[1, 1, 1][1, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None
        sub_18: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_206, getitem_112);  convert_element_type_206 = getitem_112 = None
        add_67: "f32[1, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_111, 1e-05);  getitem_111 = None
        rsqrt_18: "f32[1, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
        mul_73: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = rsqrt_18 = None
        mul_74: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_73, arg143_1);  mul_73 = arg143_1 = None
        add_68: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_74, arg144_1);  mul_74 = arg144_1 = None
        convert_element_type_207: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_68, torch.bfloat16);  add_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        view_148: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_207, [1, 384])
        permute_95: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        addmm_44: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg146_1, view_148, permute_95);  arg146_1 = view_148 = permute_95 = None
        view_149: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_44, [1, 1, 384]);  addmm_44 = None
        mul_75: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_149, 0.125);  view_149 = None
        view_150: "bf16[1, 1, 6, 64][384, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_75, [1, 1, -1, 64]);  mul_75 = None
        permute_96: "bf16[1, 6, 1, 64][384, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_150, [0, 2, 1, 3]);  view_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_151: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_207, [1, 384])
        permute_97: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
        mm_10: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_151, permute_97);  view_151 = permute_97 = None
        view_152: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_10, [1, 1, 384]);  mm_10 = None
        view_153: "bf16[1, 1, 6, 64][384, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_152, [1, -1, 6, 64]);  view_152 = None
        permute_98: "bf16[1, 6, 1, 64][384, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_153, [0, 2, 1, 3]);  view_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_154: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_207, [1, 384]);  convert_element_type_207 = None
        permute_99: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
        addmm_45: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg149_1, view_154, permute_99);  arg149_1 = view_154 = permute_99 = None
        view_155: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_45, [1, 1, 384]);  addmm_45 = None
        view_156: "bf16[1, 1, 6, 64][384, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_155, [1, -1, 6, 64]);  view_155 = None
        permute_100: "bf16[1, 6, 1, 64][384, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        scalar_tensor_7: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_6: "bf16[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-inf, dtype = torch.bfloat16, device = device(type='cuda', index=0))
        where_3: "bf16[1, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.where.self(expand, scalar_tensor_7, scalar_tensor_6);  expand = scalar_tensor_7 = scalar_tensor_6 = None
        constant_pad_nd_3: "bf16[1, 1, 1, 8][8, 8, 8, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(where_3, [0, 7], 0.0);  where_3 = None
        slice_4: "bf16[1, 1, 1, 1][8, 8, 8, 1]cuda:0" = torch.ops.aten.slice.Tensor(constant_pad_nd_3, -1, 0, 1);  constant_pad_nd_3 = None
        expand_4: "bf16[1, 6, 1, 1][8, 0, 8, 1]cuda:0" = torch.ops.aten.expand.default(slice_4, [1, 6, 1, 1]);  slice_4 = None
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_96, permute_98, permute_100, expand_4, False, scale = 1.0);  permute_96 = expand_4 = None
        getitem_113: "bf16[1, 6, 1, 64][384, 64, 384, 1]cuda:0" = _scaled_dot_product_efficient_attention_3[0];  _scaled_dot_product_efficient_attention_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_101: "bf16[1, 1, 6, 64][384, 384, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_113, [0, 2, 1, 3]);  getitem_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:353 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_157: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_101, [1, 1, -1]);  permute_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        view_158: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_157, [1, 384]);  view_157 = None
        permute_102: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
        addmm_46: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg151_1, view_158, permute_102);  arg151_1 = view_158 = permute_102 = None
        view_159: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_46, [1, 1, 384]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:480 in forward, code: hidden_states = residual + hidden_states
        add_69: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_66, view_159);  add_66 = view_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:485 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        convert_element_type_219: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_69, torch.float32)
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_219, [2], correction = 0, keepdim = True)
        getitem_117: "f32[1, 1, 1][1, 1, 1]cuda:0" = var_mean_19[0]
        getitem_118: "f32[1, 1, 1][1, 1, 1]cuda:0" = var_mean_19[1];  var_mean_19 = None
        sub_19: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_219, getitem_118);  convert_element_type_219 = getitem_118 = None
        add_70: "f32[1, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_117, 1e-05);  getitem_117 = None
        rsqrt_19: "f32[1, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
        mul_76: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = rsqrt_19 = None
        mul_77: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_76, arg152_1);  mul_76 = arg152_1 = None
        add_71: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_77, arg153_1);  mul_77 = arg153_1 = None
        convert_element_type_220: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_71, torch.bfloat16);  add_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        view_160: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_220, [1, 384]);  convert_element_type_220 = None
        permute_103: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
        addmm_47: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg155_1, view_160, permute_103);  arg155_1 = view_160 = permute_103 = None
        view_161: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_47, [1, 1, 384]);  addmm_47 = None
        mul_78: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_161, 0.125);  view_161 = None
        view_162: "bf16[1, 1, 6, 64][384, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mul_78, [1, 1, -1, 64]);  mul_78 = None
        permute_104: "bf16[1, 6, 1, 64][384, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_163: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_97, [1500, 384])
        permute_105: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
        mm_11: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_163, permute_105);  view_163 = permute_105 = None
        view_164: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_11, [1, 1500, 384]);  mm_11 = None
        view_165: "bf16[1, 1500, 6, 64][576000, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_164, [1, -1, 6, 64]);  view_164 = None
        permute_106: "bf16[1, 6, 1500, 64][576000, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_165, [0, 2, 1, 3]);  view_165 = None
        clone_72: "bf16[1, 6, 1500, 64][576000, 96000, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_166: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_97, [1500, 384])
        permute_107: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
        addmm_48: "bf16[1500, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg158_1, view_166, permute_107);  arg158_1 = view_166 = permute_107 = None
        view_167: "bf16[1, 1500, 384][576000, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_48, [1, 1500, 384]);  addmm_48 = None
        view_168: "bf16[1, 1500, 6, 64][576000, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_167, [1, -1, 6, 64]);  view_167 = None
        permute_108: "bf16[1, 6, 1500, 64][576000, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
        clone_73: "bf16[1, 6, 1500, 64][576000, 96000, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_108, memory_format = torch.contiguous_format);  permute_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_7 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_104, clone_72, clone_73, None, False, scale = 1.0);  permute_104 = None
        getitem_119: "bf16[1, 6, 1, 64][384, 64, 64, 1]cuda:0" = _scaled_dot_product_cudnn_attention_7[0];  _scaled_dot_product_cudnn_attention_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_109: "bf16[1, 1, 6, 64][384, 64, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_119, [0, 2, 1, 3]);  getitem_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:353 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_169: "bf16[1, 1, 384][384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_109, [1, 1, -1]);  permute_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        view_170: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_169, [1, 384]);  view_169 = None
        permute_110: "bf16[384, 384][1, 384]cuda:0" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        addmm_49: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg160_1, view_170, permute_110);  arg160_1 = view_170 = permute_110 = None
        view_171: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_49, [1, 1, 384]);  addmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:494 in forward, code: hidden_states = residual + hidden_states
        add_72: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_69, view_171);  add_69 = view_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:498 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_232: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_72, torch.float32)
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_232, [2], correction = 0, keepdim = True)
        getitem_128: "f32[1, 1, 1][1, 1, 1]cuda:0" = var_mean_20[0]
        getitem_129: "f32[1, 1, 1][1, 1, 1]cuda:0" = var_mean_20[1];  var_mean_20 = None
        sub_20: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_232, getitem_129);  convert_element_type_232 = getitem_129 = None
        add_73: "f32[1, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_128, 1e-05);  getitem_128 = None
        rsqrt_20: "f32[1, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
        mul_79: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = rsqrt_20 = None
        mul_80: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_79, arg161_1);  mul_79 = arg161_1 = None
        add_74: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_80, arg162_1);  mul_80 = arg162_1 = None
        convert_element_type_233: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_74, torch.bfloat16);  add_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:499 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_172: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_233, [1, 384]);  convert_element_type_233 = None
        permute_111: "bf16[384, 1536][1, 384]cuda:0" = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
        addmm_50: "bf16[1, 1536][1536, 1]cuda:0" = torch.ops.aten.addmm.default(arg164_1, view_172, permute_111);  arg164_1 = view_172 = permute_111 = None
        view_173: "bf16[1, 1, 1536][1536, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_50, [1, 1, 1536]);  addmm_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_237: "f32[1, 1, 1536][1536, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_173, torch.float32);  view_173 = None
        mul_81: "f32[1, 1, 1536][1536, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_237, 0.5)
        mul_82: "f32[1, 1, 1536][1536, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_237, 0.7071067811865476);  convert_element_type_237 = None
        erf_9: "f32[1, 1, 1536][1536, 1536, 1]cuda:0" = torch.ops.aten.erf.default(mul_82);  mul_82 = None
        add_75: "f32[1, 1, 1536][1536, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_83: "f32[1, 1, 1536][1536, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_81, add_75);  mul_81 = add_75 = None
        convert_element_type_238: "bf16[1, 1, 1536][1536, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_83, torch.bfloat16);  mul_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:501 in forward, code: hidden_states = self.fc2(hidden_states)
        view_174: "bf16[1, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_238, [1, 1536]);  convert_element_type_238 = None
        permute_112: "bf16[1536, 384][1, 1536]cuda:0" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        addmm_51: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.addmm.default(arg166_1, view_174, permute_112);  arg166_1 = view_174 = permute_112 = None
        view_175: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_51, [1, 1, 384]);  addmm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:503 in forward, code: hidden_states = residual + hidden_states
        add_76: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_72, view_175);  add_72 = view_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:790 in forward, code: hidden_states = self.layer_norm(hidden_states)
        convert_element_type_242: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_76, torch.float32);  add_76 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_242, [2], correction = 0, keepdim = True)
        getitem_130: "f32[1, 1, 1][1, 1, 1]cuda:0" = var_mean_21[0]
        getitem_131: "f32[1, 1, 1][1, 1, 1]cuda:0" = var_mean_21[1];  var_mean_21 = None
        sub_21: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_242, getitem_131);  convert_element_type_242 = getitem_131 = None
        add_77: "f32[1, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
        rsqrt_21: "f32[1, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
        mul_84: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = rsqrt_21 = None
        mul_85: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_84, arg167_1);  mul_84 = arg167_1 = None
        add_78: "f32[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_85, arg168_1);  mul_85 = arg168_1 = None
        convert_element_type_243: "bf16[1, 1, 384][384, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_78, torch.bfloat16);  add_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:1080 in forward, code: lm_logits = self.proj_out(outputs.last_hidden_state)
        view_176: "bf16[1, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_243, [1, 384]);  convert_element_type_243 = None
        permute_113: "bf16[384, 51865][1, 384]cuda:0" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        mm_12: "bf16[1, 51865][51865, 1]cuda:0" = torch.ops.aten.mm.default(view_176, permute_113);  view_176 = permute_113 = None
        view_177: "bf16[1, 1, 51865][51865, 51865, 1]cuda:0" = torch.ops.aten.reshape.default(mm_12, [1, 1, 51865]);  mm_12 = None
        return (permute_46, permute_44, permute_64, permute_62, permute_82, permute_80, permute_100, permute_98, clone_43, clone_42, clone_53, clone_52, clone_63, clone_62, clone_73, clone_72, view_177, convert_element_type_97)

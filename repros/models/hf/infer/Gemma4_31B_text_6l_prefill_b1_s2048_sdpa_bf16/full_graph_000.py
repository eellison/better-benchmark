class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[1, 2048][2048, 1]cuda:0", arg1_1: "bf16[262144, 5376][5376, 1]cuda:0", arg2_1: "bf16[][]cuda:0", arg3_1: "f32[256][1]cuda:0", arg4_1: "f32[128][1]cuda:0", arg5_1: "bf16[5376][1]cuda:0", arg6_1: "bf16[8192, 5376][5376, 1]cuda:0", arg7_1: "bf16[256][1]cuda:0", arg8_1: "bf16[4096, 5376][5376, 1]cuda:0", arg9_1: "bf16[4096, 5376][5376, 1]cuda:0", arg10_1: "bf16[256][1]cuda:0", arg11_1: "bf16[5376, 8192][8192, 1]cuda:0", arg12_1: "bf16[5376][1]cuda:0", arg13_1: "bf16[5376][1]cuda:0", arg14_1: "bf16[21504, 5376][5376, 1]cuda:0", arg15_1: "bf16[21504, 5376][5376, 1]cuda:0", arg16_1: "bf16[5376, 21504][21504, 1]cuda:0", arg17_1: "bf16[5376][1]cuda:0", arg18_1: "bf16[1][1]cuda:0", arg19_1: "bf16[5376][1]cuda:0", arg20_1: "bf16[8192, 5376][5376, 1]cuda:0", arg21_1: "bf16[256][1]cuda:0", arg22_1: "bf16[4096, 5376][5376, 1]cuda:0", arg23_1: "bf16[4096, 5376][5376, 1]cuda:0", arg24_1: "bf16[256][1]cuda:0", arg25_1: "bf16[5376, 8192][8192, 1]cuda:0", arg26_1: "bf16[5376][1]cuda:0", arg27_1: "bf16[5376][1]cuda:0", arg28_1: "bf16[21504, 5376][5376, 1]cuda:0", arg29_1: "bf16[21504, 5376][5376, 1]cuda:0", arg30_1: "bf16[5376, 21504][21504, 1]cuda:0", arg31_1: "bf16[5376][1]cuda:0", arg32_1: "bf16[1][1]cuda:0", arg33_1: "bf16[5376][1]cuda:0", arg34_1: "bf16[8192, 5376][5376, 1]cuda:0", arg35_1: "bf16[256][1]cuda:0", arg36_1: "bf16[4096, 5376][5376, 1]cuda:0", arg37_1: "bf16[4096, 5376][5376, 1]cuda:0", arg38_1: "bf16[256][1]cuda:0", arg39_1: "bf16[5376, 8192][8192, 1]cuda:0", arg40_1: "bf16[5376][1]cuda:0", arg41_1: "bf16[5376][1]cuda:0", arg42_1: "bf16[21504, 5376][5376, 1]cuda:0", arg43_1: "bf16[21504, 5376][5376, 1]cuda:0", arg44_1: "bf16[5376, 21504][21504, 1]cuda:0", arg45_1: "bf16[5376][1]cuda:0", arg46_1: "bf16[1][1]cuda:0", arg47_1: "bf16[5376][1]cuda:0", arg48_1: "bf16[8192, 5376][5376, 1]cuda:0", arg49_1: "bf16[256][1]cuda:0", arg50_1: "bf16[4096, 5376][5376, 1]cuda:0", arg51_1: "bf16[4096, 5376][5376, 1]cuda:0", arg52_1: "bf16[256][1]cuda:0", arg53_1: "bf16[5376, 8192][8192, 1]cuda:0", arg54_1: "bf16[5376][1]cuda:0", arg55_1: "bf16[5376][1]cuda:0", arg56_1: "bf16[21504, 5376][5376, 1]cuda:0", arg57_1: "bf16[21504, 5376][5376, 1]cuda:0", arg58_1: "bf16[5376, 21504][21504, 1]cuda:0", arg59_1: "bf16[5376][1]cuda:0", arg60_1: "bf16[1][1]cuda:0", arg61_1: "bf16[5376][1]cuda:0", arg62_1: "bf16[8192, 5376][5376, 1]cuda:0", arg63_1: "bf16[256][1]cuda:0", arg64_1: "bf16[4096, 5376][5376, 1]cuda:0", arg65_1: "bf16[4096, 5376][5376, 1]cuda:0", arg66_1: "bf16[256][1]cuda:0", arg67_1: "bf16[5376, 8192][8192, 1]cuda:0", arg68_1: "bf16[5376][1]cuda:0", arg69_1: "bf16[5376][1]cuda:0", arg70_1: "bf16[21504, 5376][5376, 1]cuda:0", arg71_1: "bf16[21504, 5376][5376, 1]cuda:0", arg72_1: "bf16[5376, 21504][21504, 1]cuda:0", arg73_1: "bf16[5376][1]cuda:0", arg74_1: "bf16[1][1]cuda:0", arg75_1: "bf16[5376][1]cuda:0", arg76_1: "bf16[16384, 5376][5376, 1]cuda:0", arg77_1: "bf16[512][1]cuda:0", arg78_1: "bf16[2048, 5376][5376, 1]cuda:0", arg79_1: "bf16[512][1]cuda:0", arg80_1: "bf16[5376, 16384][16384, 1]cuda:0", arg81_1: "bf16[5376][1]cuda:0", arg82_1: "bf16[5376][1]cuda:0", arg83_1: "bf16[21504, 5376][5376, 1]cuda:0", arg84_1: "bf16[21504, 5376][5376, 1]cuda:0", arg85_1: "bf16[5376, 21504][21504, 1]cuda:0", arg86_1: "bf16[5376][1]cuda:0", arg87_1: "bf16[1][1]cuda:0", arg88_1: "bf16[5376][1]cuda:0"):
        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1444 in forward, code: return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)
        embedding: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 0);  arg0_1 = None
        mul: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(embedding, arg2_1);  embedding = arg2_1 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:220 in forward, code: normed_output = self._norm(hidden_states.float())
        convert_element_type_6: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul, torch.float32)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:215 in _norm, code: mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        pow_1: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_6, 2)
        mean: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add_5: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:217 in _norm, code: return hidden_states * torch.pow(mean_squared, -0.5)
        pow_2: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_5, -0.5);  add_5 = None
        mul_7: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_6, pow_2);  convert_element_type_6 = pow_2 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:222 in forward, code: normed_output = normed_output * self.weight.float()
        convert_element_type_7: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg5_1, torch.float32);  arg5_1 = None
        mul_8: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_7, convert_element_type_7);  mul_7 = convert_element_type_7 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:223 in forward, code: return normed_output.type_as(hidden_states)
        convert_element_type_8: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_8, torch.bfloat16);  mul_8 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1217 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape)
        view_2: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_8, [2048, 5376])
        permute: "bf16[5376, 8192][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        mm: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_2, permute);  view_2 = permute = None
        view_3: "bf16[1, 2048, 8192][16777216, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [1, 2048, 8192]);  mm = None
        view_4: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_3, [1, 2048, -1, 256]);  view_3 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:220 in forward, code: normed_output = self._norm(hidden_states.float())
        convert_element_type_11: "f32[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_4, torch.float32);  view_4 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:215 in _norm, code: mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        pow_3: "f32[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_11, 2)
        mean_1: "f32[1, 2048, 32, 1][65536, 32, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_3, [-1], True);  pow_3 = None
        add_6: "f32[1, 2048, 32, 1][65536, 32, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:217 in _norm, code: return hidden_states * torch.pow(mean_squared, -0.5)
        pow_4: "f32[1, 2048, 32, 1][65536, 32, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_6, -0.5);  add_6 = None
        mul_9: "f32[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_11, pow_4);  convert_element_type_11 = pow_4 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:222 in forward, code: normed_output = normed_output * self.weight.float()
        convert_element_type_12: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg7_1, torch.float32);  arg7_1 = None
        mul_10: "f32[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_9, convert_element_type_12);  mul_9 = convert_element_type_12 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:223 in forward, code: return normed_output.type_as(hidden_states)
        convert_element_type_13: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_10, torch.bfloat16);  mul_10 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1647 in forward, code: position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        iota: "i64[2048][1]cuda:0" = torch.ops.prims.iota.default(2048, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1648 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze: "i64[1, 2048][2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1143 in forward, code: freqs = position_ids[..., None].float() * inv_freq.to(device=x.device, dtype=torch.float)
        unsqueeze_21: "i64[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, 2)
        convert_element_type_3: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(unsqueeze_21, torch.float32);  unsqueeze_21 = None
        mul_4: "f32[1, 2048, 128][262144, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_3, arg4_1);  convert_element_type_3 = arg4_1 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1144 in forward, code: emb = torch.cat((freqs, freqs), dim=-1)
        unsqueeze_22: "f32[1, 2048, 1, 128][262144, 128, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4, 2);  mul_4 = None
        expand_3: "f32[1, 2048, 2, 128][262144, 128, 0, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_22, [1, 2048, 2, 128]);  unsqueeze_22 = None
        clone_2: "f32[1, 2048, 2, 128][524288, 256, 128, 1]cuda:0" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_1: "f32[1, 2048, 256][524288, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_2, [1, 2048, 256]);  clone_2 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1145 in forward, code: cos = emb.cos() * attention_scaling
        cos_1: "f32[1, 2048, 256][524288, 256, 1]cuda:0" = torch.ops.aten.cos.default(view_1)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1148 in forward, code: return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        convert_element_type_4: "bf16[1, 2048, 256][524288, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cos_1, torch.bfloat16);  cos_1 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:796 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_23: "bf16[1, 2048, 1, 256][524288, 256, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_4, 2)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:798 in apply_rotary_pos_emb, code: return (x * cos) + (rotate_half(x) * sin)
        mul_11: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_13, unsqueeze_23);  unsqueeze_23 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:775 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_8: "bf16[1, 2048, 32, 128][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_13, 3, 128, 9223372036854775807)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:776 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg: "bf16[1, 2048, 32, 128][8388608, 4096, 128, 1]cuda:0" = torch.ops.aten.neg.default(slice_8);  slice_8 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:774 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_7: "bf16[1, 2048, 32, 128][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_13, 3, 0, 128);  convert_element_type_13 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:776 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_2: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg, slice_7], -1);  neg = slice_7 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1146 in forward, code: sin = emb.sin() * attention_scaling
        sin_1: "f32[1, 2048, 256][524288, 256, 1]cuda:0" = torch.ops.aten.sin.default(view_1);  view_1 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1148 in forward, code: return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        convert_element_type_5: "bf16[1, 2048, 256][524288, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sin_1, torch.bfloat16);  sin_1 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:797 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_24: "bf16[1, 2048, 1, 256][524288, 256, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_5, 2)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:798 in apply_rotary_pos_emb, code: return (x * cos) + (rotate_half(x) * sin)
        mul_12: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_2, unsqueeze_24);  cat_2 = unsqueeze_24 = None
        add_7: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1220 in forward, code: query_states = query_states.transpose(1, 2)
        permute_1: "bf16[1, 32, 2048, 256][16777216, 256, 8192, 1]cuda:0" = torch.ops.aten.permute.default(add_7, [0, 2, 1, 3]);  add_7 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1231 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape)
        view_5: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_8, [2048, 5376])
        permute_2: "bf16[5376, 4096][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        mm_1: "bf16[2048, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_5, permute_2);  view_5 = permute_2 = None
        view_6: "bf16[1, 2048, 4096][8388608, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_1, [1, 2048, 4096]);  mm_1 = None
        view_7: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_6, [1, 2048, -1, 256]);  view_6 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:220 in forward, code: normed_output = self._norm(hidden_states.float())
        convert_element_type_18: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_7, torch.float32);  view_7 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:215 in _norm, code: mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        pow_5: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_18, 2)
        mean_2: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_5, [-1], True);  pow_5 = None
        add_8: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_2, 1e-06);  mean_2 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:217 in _norm, code: return hidden_states * torch.pow(mean_squared, -0.5)
        pow_6: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_8, -0.5);  add_8 = None
        mul_13: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_18, pow_6);  convert_element_type_18 = pow_6 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:222 in forward, code: normed_output = normed_output * self.weight.float()
        convert_element_type_19: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg10_1, torch.float32);  arg10_1 = None
        mul_14: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_13, convert_element_type_19);  mul_13 = convert_element_type_19 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:223 in forward, code: return normed_output.type_as(hidden_states)
        convert_element_type_20: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_14, torch.bfloat16);  mul_14 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:796 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_25: "bf16[1, 2048, 1, 256][524288, 256, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_4, 2)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:798 in apply_rotary_pos_emb, code: return (x * cos) + (rotate_half(x) * sin)
        mul_15: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_20, unsqueeze_25);  unsqueeze_25 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:775 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_10: "bf16[1, 2048, 16, 128][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_20, 3, 128, 9223372036854775807)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:776 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_1: "bf16[1, 2048, 16, 128][4194304, 2048, 128, 1]cuda:0" = torch.ops.aten.neg.default(slice_10);  slice_10 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:774 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_9: "bf16[1, 2048, 16, 128][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_20, 3, 0, 128);  convert_element_type_20 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:776 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_3: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_1, slice_9], -1);  neg_1 = slice_9 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:797 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_26: "bf16[1, 2048, 1, 256][524288, 256, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_5, 2)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:798 in apply_rotary_pos_emb, code: return (x * cos) + (rotate_half(x) * sin)
        mul_16: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_3, unsqueeze_26);  cat_3 = unsqueeze_26 = None
        add_9: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1236 in forward, code: key_states = key_states.transpose(1, 2)
        permute_4: "bf16[1, 16, 2048, 256][8388608, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(add_9, [0, 2, 1, 3]);  add_9 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/integrations/sdpa_attention.py:26 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_27: "bf16[1, 16, 1, 2048, 256][8388608, 256, 8388608, 4096, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_4, 2);  permute_4 = None
        expand_4: "bf16[1, 16, 2, 2048, 256][8388608, 256, 0, 4096, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_27, [1, 16, 2, 2048, 256]);  unsqueeze_27 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/integrations/sdpa_attention.py:27 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_4: "bf16[1, 16, 2, 2048, 256][16777216, 1048576, 524288, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_11: "bf16[1, 32, 2048, 256][16777216, 524288, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_4, [1, 32, 2048, 256]);  clone_4 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1232 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states
        view_8: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_8, [2048, 5376]);  convert_element_type_8 = None
        permute_3: "bf16[5376, 4096][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        mm_2: "bf16[2048, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_8, permute_3);  view_8 = permute_3 = None
        view_9: "bf16[1, 2048, 4096][8388608, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_2, [1, 2048, 4096]);  mm_2 = None
        view_10: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_9, [1, 2048, -1, 256]);  view_9 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:220 in forward, code: normed_output = self._norm(hidden_states.float())
        convert_element_type_21: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_10, torch.float32);  view_10 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:215 in _norm, code: mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        pow_7: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_21, 2)
        mean_3: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_7, [-1], True);  pow_7 = None
        add_10: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_3, 1e-06);  mean_3 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:217 in _norm, code: return hidden_states * torch.pow(mean_squared, -0.5)
        pow_8: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_10, -0.5);  add_10 = None
        mul_17: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_21, pow_8);  convert_element_type_21 = pow_8 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:223 in forward, code: return normed_output.type_as(hidden_states)
        convert_element_type_22: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_17, torch.bfloat16);  mul_17 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1239 in forward, code: value_states = value_states.transpose(1, 2)
        permute_5: "bf16[1, 16, 2048, 256][8388608, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_22, [0, 2, 1, 3]);  convert_element_type_22 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/integrations/sdpa_attention.py:26 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_28: "bf16[1, 16, 1, 2048, 256][8388608, 256, 8388608, 4096, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_5, 2);  permute_5 = None
        expand_5: "bf16[1, 16, 2, 2048, 256][8388608, 256, 0, 4096, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_28, [1, 16, 2, 2048, 256]);  unsqueeze_28 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/integrations/sdpa_attention.py:27 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_5: "bf16[1, 16, 2, 2048, 256][16777216, 1048576, 524288, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_12: "bf16[1, 32, 2048, 256][16777216, 524288, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_5, [1, 32, 2048, 256]);  clone_5 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/masking_utils.py:50 in and_mask, code: result = q_idx.new_ones((), dtype=torch.bool)
        full_default_1: "b8[][]cuda:0" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "b8[][]cuda:0" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/masking_utils.py:510 in sdpa_mask, code: kv_arange = torch.arange(kv_length, device=device) + kv_offset
        iota_8: "i64[2048][1]cuda:0" = torch.ops.prims.iota.default(2048, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/masking_utils.py:370 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_16: "i64[1, 2048][2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_8, 0);  iota_8 = None
        unsqueeze_17: "i64[1, 1, 2048][2048, 2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_16, 1);  unsqueeze_16 = None
        unsqueeze_18: "i64[1, 1, 1, 2048][2048, 2048, 2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_17, 2);  unsqueeze_17 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/masking_utils.py:509 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_7: "i64[2048][1]cuda:0" = torch.ops.prims.iota.default(2048, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/masking_utils.py:369 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_13: "i64[1, 2048][2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_7, 0);  iota_7 = None
        unsqueeze_14: "i64[1, 1, 2048][2048, 2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_13, 1);  unsqueeze_13 = None
        unsqueeze_15: "i64[1, 1, 2048, 1][2048, 2048, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_14, 3);  unsqueeze_14 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/masking_utils.py:95 in inner_mask, code: return kv_idx > q_idx - sliding_window
        sub_4: "i64[1, 1, 2048, 1][2048, 2048, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(unsqueeze_15, 1024)
        gt: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.gt.Tensor(unsqueeze_18, sub_4);  sub_4 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/masking_utils.py:52 in and_mask, code: result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        bitwise_and_2: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(full_default_2, gt);  full_default_2 = gt = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/masking_utils.py:76 in causal_mask_function, code: return kv_idx <= q_idx
        le_1: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.le.Tensor(unsqueeze_18, unsqueeze_15)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/masking_utils.py:52 in and_mask, code: result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        bitwise_and_3: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_2, le_1);  bitwise_and_2 = le_1 = None
        bitwise_and_4: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(full_default_1, bitwise_and_3);  full_default_1 = bitwise_and_3 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/masking_utils.py:735 in find_packed_sequence_indices, code: first_dummy_value = position_ids[:, :1] - 1  # We just need the diff on this first value to be 1
        slice_4: "i64[1, 1][2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(unsqueeze, 1, 0, 1)
        sub_2: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.sub.Tensor(slice_4, 1);  slice_4 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/masking_utils.py:736 in find_packed_sequence_indices, code: position_diff = torch.diff(position_ids, prepend=first_dummy_value, dim=-1)
        cat_1: "i64[1, 2049][2049, 1]cuda:0" = torch.ops.aten.cat.default([sub_2, unsqueeze], -1);  sub_2 = None
        slice_6: "i64[1, 2048][2049, 1]cuda:0" = torch.ops.aten.slice.Tensor(cat_1, -1, 1, 2049)
        slice_5: "i64[1, 2048][2049, 1]cuda:0" = torch.ops.aten.slice.Tensor(cat_1, -1, 0, 2048);  cat_1 = None
        sub_3: "i64[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(slice_6, slice_5);  slice_6 = slice_5 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/masking_utils.py:737 in find_packed_sequence_indices, code: packed_sequence_mask = (position_diff != 1).cumsum(-1)
        ne_1: "b8[1, 2048][2048, 1]cuda:0" = torch.ops.aten.ne.Scalar(sub_3, 1);  sub_3 = None
        cumsum_1: "i64[1, 2048][2048, 1]cuda:0" = torch.ops.aten.cumsum.default(ne_1, -1);  ne_1 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/masking_utils.py:507 in sdpa_mask, code: batch_arange = torch.arange(batch_size, device=device)
        iota_5: "i64[1][1]cuda:0" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/masking_utils.py:367 in _non_vmap_expansion_sdpa, code: batch_indices = batch_indices[:, None, None, None]
        unsqueeze_10: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_5, 1);  iota_5 = None
        unsqueeze_11: "i64[1, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 2);  unsqueeze_10 = None
        unsqueeze_12: "i64[1, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_11, 3);  unsqueeze_11 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/masking_utils.py:184 in inner_mask, code: return packed_sequence_mask[batch_idx, q_idx] == packed_sequence_mask[batch_idx, kv_idx]
        index_2: "i64[1, 1, 2048, 1][2048, 2048, 1, 1]cuda:0" = torch.ops.aten.index.Tensor(cumsum_1, [unsqueeze_12, unsqueeze_15]);  unsqueeze_15 = None
        index_3: "i64[1, 1, 1, 2048][2048, 2048, 2048, 1]cuda:0" = torch.ops.aten.index.Tensor(cumsum_1, [unsqueeze_12, unsqueeze_18]);  cumsum_1 = unsqueeze_12 = unsqueeze_18 = None
        eq_1: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.eq.Tensor(index_2, index_3);  index_2 = index_3 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/masking_utils.py:52 in and_mask, code: result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        bitwise_and_5: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_4, eq_1);  bitwise_and_4 = eq_1 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/masking_utils.py:518 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_1: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.expand.default(bitwise_and_5, [1, -1, 2048, 2048]);  bitwise_and_5 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/integrations/sdpa_attention.py:174 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_4: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_3: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.where.self(expand_1, full_default_4, full_default_3);  full_default_4 = full_default_3 = None
        expand_6: "bf16[1, 32, 2048, 2048][4194304, 0, 2048, 1]cuda:0" = torch.ops.aten.expand.default(where, [1, 32, 2048, 2048]);  where = None
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_1, view_11, view_12, expand_6, False, scale = 1.0);  permute_1 = view_11 = view_12 = expand_6 = None
        getitem: "bf16[1, 32, 2048, 256][16777216, 256, 8192, 1]cuda:0" = _scaled_dot_product_efficient_attention[0];  _scaled_dot_product_efficient_attention = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/integrations/sdpa_attention.py:184 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_6: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.permute.default(getitem, [0, 2, 1, 3]);  getitem = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1262 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_13: "bf16[1, 2048, 8192][16777216, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(permute_6, [1, 2048, -1]);  permute_6 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1263 in forward, code: attn_output = self.o_proj(attn_output)
        view_14: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(view_13, [2048, 8192]);  view_13 = None
        permute_7: "bf16[8192, 5376][1, 8192]cuda:0" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        mm_3: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.mm.default(view_14, permute_7);  view_14 = permute_7 = None
        view_15: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.reshape.default(mm_3, [1, 2048, 5376]);  mm_3 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:220 in forward, code: normed_output = self._norm(hidden_states.float())
        convert_element_type_25: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_15, torch.float32);  view_15 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:215 in _norm, code: mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        pow_9: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_25, 2)
        mean_4: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_9, [-1], True);  pow_9 = None
        add_11: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_4, 1e-06);  mean_4 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:217 in _norm, code: return hidden_states * torch.pow(mean_squared, -0.5)
        pow_10: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_11, -0.5);  add_11 = None
        mul_18: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_25, pow_10);  convert_element_type_25 = pow_10 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:222 in forward, code: normed_output = normed_output * self.weight.float()
        convert_element_type_26: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg12_1, torch.float32);  arg12_1 = None
        mul_19: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_18, convert_element_type_26);  mul_18 = convert_element_type_26 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:223 in forward, code: return normed_output.type_as(hidden_states)
        convert_element_type_27: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_19, torch.bfloat16);  mul_19 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1397 in forward, code: hidden_states = residual + hidden_states
        add_12: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.add.Tensor(mul, convert_element_type_27);  mul = convert_element_type_27 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:220 in forward, code: normed_output = self._norm(hidden_states.float())
        convert_element_type_28: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_12, torch.float32)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:215 in _norm, code: mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        pow_11: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_28, 2)
        mean_5: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_11, [-1], True);  pow_11 = None
        add_13: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_5, 1e-06);  mean_5 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:217 in _norm, code: return hidden_states * torch.pow(mean_squared, -0.5)
        pow_12: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_13, -0.5);  add_13 = None
        mul_20: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_28, pow_12);  convert_element_type_28 = pow_12 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:222 in forward, code: normed_output = normed_output * self.weight.float()
        convert_element_type_29: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg13_1, torch.float32);  arg13_1 = None
        mul_21: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_20, convert_element_type_29);  mul_20 = convert_element_type_29 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:223 in forward, code: return normed_output.type_as(hidden_states)
        convert_element_type_30: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_21, torch.bfloat16);  mul_21 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1075 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_16: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_30, [2048, 5376])
        permute_8: "bf16[5376, 21504][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
        mm_4: "bf16[2048, 21504][21504, 1]cuda:0" = torch.ops.aten.mm.default(view_16, permute_8);  view_16 = permute_8 = None
        view_17: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.reshape.default(mm_4, [1, 2048, 21504]);  mm_4 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/activations.py:51 in forward, code: return self.act(input)
        convert_element_type_33: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_17, torch.float32);  view_17 = None
        mul_26: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_33, 0.5)
        mul_22: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_33, convert_element_type_33)
        mul_23: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_22, convert_element_type_33);  mul_22 = None
        mul_24: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_23, 0.044715);  mul_23 = None
        add_14: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_33, mul_24);  convert_element_type_33 = mul_24 = None
        mul_25: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_14, 0.7978845608028654);  add_14 = None
        tanh: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.tanh.default(mul_25);  mul_25 = None
        add_15: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh, 1);  tanh = None
        mul_27: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_26, add_15);  mul_26 = add_15 = None
        convert_element_type_34: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_27, torch.bfloat16);  mul_27 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1075 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_18: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_30, [2048, 5376]);  convert_element_type_30 = None
        permute_9: "bf16[5376, 21504][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        mm_5: "bf16[2048, 21504][21504, 1]cuda:0" = torch.ops.aten.mm.default(view_18, permute_9);  view_18 = permute_9 = None
        view_19: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.reshape.default(mm_5, [1, 2048, 21504]);  mm_5 = None
        mul_28: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_34, view_19);  convert_element_type_34 = view_19 = None
        view_20: "bf16[2048, 21504][21504, 1]cuda:0" = torch.ops.aten.reshape.default(mul_28, [2048, 21504]);  mul_28 = None
        permute_10: "bf16[21504, 5376][1, 21504]cuda:0" = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
        mm_6: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.mm.default(view_20, permute_10);  view_20 = permute_10 = None
        view_21: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.reshape.default(mm_6, [1, 2048, 5376]);  mm_6 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:220 in forward, code: normed_output = self._norm(hidden_states.float())
        convert_element_type_39: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_21, torch.float32);  view_21 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:215 in _norm, code: mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        pow_13: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_39, 2)
        mean_6: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_13, [-1], True);  pow_13 = None
        add_16: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_6, 1e-06);  mean_6 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:217 in _norm, code: return hidden_states * torch.pow(mean_squared, -0.5)
        pow_14: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_16, -0.5);  add_16 = None
        mul_29: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_39, pow_14);  convert_element_type_39 = pow_14 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:222 in forward, code: normed_output = normed_output * self.weight.float()
        convert_element_type_40: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg17_1, torch.float32);  arg17_1 = None
        mul_30: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_29, convert_element_type_40);  mul_29 = convert_element_type_40 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:223 in forward, code: return normed_output.type_as(hidden_states)
        convert_element_type_41: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_30, torch.bfloat16);  mul_30 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1418 in forward, code: hidden_states = residual + hidden_states
        add_17: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.add.Tensor(add_12, convert_element_type_41);  add_12 = convert_element_type_41 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1429 in forward, code: hidden_states *= self.layer_scalar
        mul_31: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_17, arg18_1);  add_17 = arg18_1 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:220 in forward, code: normed_output = self._norm(hidden_states.float())
        convert_element_type_42: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_31, torch.float32)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:215 in _norm, code: mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        pow_15: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_42, 2)
        mean_7: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_15, [-1], True);  pow_15 = None
        add_18: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_7, 1e-06);  mean_7 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:217 in _norm, code: return hidden_states * torch.pow(mean_squared, -0.5)
        pow_16: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_18, -0.5);  add_18 = None
        mul_32: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_42, pow_16);  convert_element_type_42 = pow_16 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:222 in forward, code: normed_output = normed_output * self.weight.float()
        convert_element_type_43: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg19_1, torch.float32);  arg19_1 = None
        mul_33: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_32, convert_element_type_43);  mul_32 = convert_element_type_43 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:223 in forward, code: return normed_output.type_as(hidden_states)
        convert_element_type_44: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_33, torch.bfloat16);  mul_33 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1217 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape)
        view_22: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_44, [2048, 5376])
        permute_11: "bf16[5376, 8192][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        mm_7: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_22, permute_11);  view_22 = permute_11 = None
        view_23: "bf16[1, 2048, 8192][16777216, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_7, [1, 2048, 8192]);  mm_7 = None
        view_24: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_23, [1, 2048, -1, 256]);  view_23 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:220 in forward, code: normed_output = self._norm(hidden_states.float())
        convert_element_type_47: "f32[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_24, torch.float32);  view_24 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:215 in _norm, code: mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        pow_17: "f32[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_47, 2)
        mean_8: "f32[1, 2048, 32, 1][65536, 32, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_17, [-1], True);  pow_17 = None
        add_19: "f32[1, 2048, 32, 1][65536, 32, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_8, 1e-06);  mean_8 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:217 in _norm, code: return hidden_states * torch.pow(mean_squared, -0.5)
        pow_18: "f32[1, 2048, 32, 1][65536, 32, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_19, -0.5);  add_19 = None
        mul_34: "f32[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_47, pow_18);  convert_element_type_47 = pow_18 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:222 in forward, code: normed_output = normed_output * self.weight.float()
        convert_element_type_48: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg21_1, torch.float32);  arg21_1 = None
        mul_35: "f32[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_34, convert_element_type_48);  mul_34 = convert_element_type_48 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:223 in forward, code: return normed_output.type_as(hidden_states)
        convert_element_type_49: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_35, torch.bfloat16);  mul_35 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:796 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_29: "bf16[1, 2048, 1, 256][524288, 256, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_4, 2)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:798 in apply_rotary_pos_emb, code: return (x * cos) + (rotate_half(x) * sin)
        mul_36: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_49, unsqueeze_29);  unsqueeze_29 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:775 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_12: "bf16[1, 2048, 32, 128][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_49, 3, 128, 9223372036854775807)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:776 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_2: "bf16[1, 2048, 32, 128][8388608, 4096, 128, 1]cuda:0" = torch.ops.aten.neg.default(slice_12);  slice_12 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:774 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_11: "bf16[1, 2048, 32, 128][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_49, 3, 0, 128);  convert_element_type_49 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:776 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_4: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_2, slice_11], -1);  neg_2 = slice_11 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:797 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_30: "bf16[1, 2048, 1, 256][524288, 256, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_5, 2)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:798 in apply_rotary_pos_emb, code: return (x * cos) + (rotate_half(x) * sin)
        mul_37: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_4, unsqueeze_30);  cat_4 = unsqueeze_30 = None
        add_20: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1220 in forward, code: query_states = query_states.transpose(1, 2)
        permute_12: "bf16[1, 32, 2048, 256][16777216, 256, 8192, 1]cuda:0" = torch.ops.aten.permute.default(add_20, [0, 2, 1, 3]);  add_20 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1231 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape)
        view_25: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_44, [2048, 5376])
        permute_13: "bf16[5376, 4096][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        mm_8: "bf16[2048, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_25, permute_13);  view_25 = permute_13 = None
        view_26: "bf16[1, 2048, 4096][8388608, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_8, [1, 2048, 4096]);  mm_8 = None
        view_27: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_26, [1, 2048, -1, 256]);  view_26 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:220 in forward, code: normed_output = self._norm(hidden_states.float())
        convert_element_type_54: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_27, torch.float32);  view_27 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:215 in _norm, code: mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        pow_19: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_54, 2)
        mean_9: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_19, [-1], True);  pow_19 = None
        add_21: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_9, 1e-06);  mean_9 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:217 in _norm, code: return hidden_states * torch.pow(mean_squared, -0.5)
        pow_20: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_21, -0.5);  add_21 = None
        mul_38: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_54, pow_20);  convert_element_type_54 = pow_20 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:222 in forward, code: normed_output = normed_output * self.weight.float()
        convert_element_type_55: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg24_1, torch.float32);  arg24_1 = None
        mul_39: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_38, convert_element_type_55);  mul_38 = convert_element_type_55 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:223 in forward, code: return normed_output.type_as(hidden_states)
        convert_element_type_56: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_39, torch.bfloat16);  mul_39 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:796 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_31: "bf16[1, 2048, 1, 256][524288, 256, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_4, 2)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:798 in apply_rotary_pos_emb, code: return (x * cos) + (rotate_half(x) * sin)
        mul_40: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_56, unsqueeze_31);  unsqueeze_31 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:775 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_14: "bf16[1, 2048, 16, 128][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_56, 3, 128, 9223372036854775807)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:776 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_3: "bf16[1, 2048, 16, 128][4194304, 2048, 128, 1]cuda:0" = torch.ops.aten.neg.default(slice_14);  slice_14 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:774 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_13: "bf16[1, 2048, 16, 128][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_56, 3, 0, 128);  convert_element_type_56 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:776 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_5: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_3, slice_13], -1);  neg_3 = slice_13 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:797 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_32: "bf16[1, 2048, 1, 256][524288, 256, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_5, 2)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:798 in apply_rotary_pos_emb, code: return (x * cos) + (rotate_half(x) * sin)
        mul_41: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_5, unsqueeze_32);  cat_5 = unsqueeze_32 = None
        add_22: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_40, mul_41);  mul_40 = mul_41 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1236 in forward, code: key_states = key_states.transpose(1, 2)
        permute_15: "bf16[1, 16, 2048, 256][8388608, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(add_22, [0, 2, 1, 3]);  add_22 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/integrations/sdpa_attention.py:26 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_33: "bf16[1, 16, 1, 2048, 256][8388608, 256, 8388608, 4096, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_15, 2);  permute_15 = None
        expand_7: "bf16[1, 16, 2, 2048, 256][8388608, 256, 0, 4096, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_33, [1, 16, 2, 2048, 256]);  unsqueeze_33 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/integrations/sdpa_attention.py:27 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_6: "bf16[1, 16, 2, 2048, 256][16777216, 1048576, 524288, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_31: "bf16[1, 32, 2048, 256][16777216, 524288, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_6, [1, 32, 2048, 256]);  clone_6 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1232 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states
        view_28: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_44, [2048, 5376]);  convert_element_type_44 = None
        permute_14: "bf16[5376, 4096][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        mm_9: "bf16[2048, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_28, permute_14);  view_28 = permute_14 = None
        view_29: "bf16[1, 2048, 4096][8388608, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_9, [1, 2048, 4096]);  mm_9 = None
        view_30: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_29, [1, 2048, -1, 256]);  view_29 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:220 in forward, code: normed_output = self._norm(hidden_states.float())
        convert_element_type_57: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_30, torch.float32);  view_30 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:215 in _norm, code: mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        pow_21: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_57, 2)
        mean_10: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_21, [-1], True);  pow_21 = None
        add_23: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_10, 1e-06);  mean_10 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:217 in _norm, code: return hidden_states * torch.pow(mean_squared, -0.5)
        pow_22: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_23, -0.5);  add_23 = None
        mul_42: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_57, pow_22);  convert_element_type_57 = pow_22 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:223 in forward, code: return normed_output.type_as(hidden_states)
        convert_element_type_58: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_42, torch.bfloat16);  mul_42 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1239 in forward, code: value_states = value_states.transpose(1, 2)
        permute_16: "bf16[1, 16, 2048, 256][8388608, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_58, [0, 2, 1, 3]);  convert_element_type_58 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/integrations/sdpa_attention.py:26 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_34: "bf16[1, 16, 1, 2048, 256][8388608, 256, 8388608, 4096, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_16, 2);  permute_16 = None
        expand_8: "bf16[1, 16, 2, 2048, 256][8388608, 256, 0, 4096, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_34, [1, 16, 2, 2048, 256]);  unsqueeze_34 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/integrations/sdpa_attention.py:27 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_7: "bf16[1, 16, 2, 2048, 256][16777216, 1048576, 524288, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_32: "bf16[1, 32, 2048, 256][16777216, 524288, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_7, [1, 32, 2048, 256]);  clone_7 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/integrations/sdpa_attention.py:174 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_6: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_5: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "bf16[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.where.self(expand_1, full_default_6, full_default_5);  full_default_6 = full_default_5 = None
        expand_9: "bf16[1, 32, 2048, 2048][4194304, 0, 2048, 1]cuda:0" = torch.ops.aten.expand.default(where_1, [1, 32, 2048, 2048]);  where_1 = None
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_12, view_31, view_32, expand_9, False, scale = 1.0);  permute_12 = view_31 = view_32 = expand_9 = None
        getitem_4: "bf16[1, 32, 2048, 256][16777216, 256, 8192, 1]cuda:0" = _scaled_dot_product_efficient_attention_1[0];  _scaled_dot_product_efficient_attention_1 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/integrations/sdpa_attention.py:184 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_17: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.permute.default(getitem_4, [0, 2, 1, 3]);  getitem_4 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1262 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_33: "bf16[1, 2048, 8192][16777216, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(permute_17, [1, 2048, -1]);  permute_17 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1263 in forward, code: attn_output = self.o_proj(attn_output)
        view_34: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(view_33, [2048, 8192]);  view_33 = None
        permute_18: "bf16[8192, 5376][1, 8192]cuda:0" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        mm_10: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.mm.default(view_34, permute_18);  view_34 = permute_18 = None
        view_35: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.reshape.default(mm_10, [1, 2048, 5376]);  mm_10 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:220 in forward, code: normed_output = self._norm(hidden_states.float())
        convert_element_type_61: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_35, torch.float32);  view_35 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:215 in _norm, code: mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        pow_23: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_61, 2)
        mean_11: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_23, [-1], True);  pow_23 = None
        add_24: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_11, 1e-06);  mean_11 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:217 in _norm, code: return hidden_states * torch.pow(mean_squared, -0.5)
        pow_24: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_24, -0.5);  add_24 = None
        mul_43: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_61, pow_24);  convert_element_type_61 = pow_24 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:222 in forward, code: normed_output = normed_output * self.weight.float()
        convert_element_type_62: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg26_1, torch.float32);  arg26_1 = None
        mul_44: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_43, convert_element_type_62);  mul_43 = convert_element_type_62 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:223 in forward, code: return normed_output.type_as(hidden_states)
        convert_element_type_63: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_44, torch.bfloat16);  mul_44 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1397 in forward, code: hidden_states = residual + hidden_states
        add_25: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_31, convert_element_type_63);  mul_31 = convert_element_type_63 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:220 in forward, code: normed_output = self._norm(hidden_states.float())
        convert_element_type_64: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_25, torch.float32)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:215 in _norm, code: mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        pow_25: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_64, 2)
        mean_12: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_25, [-1], True);  pow_25 = None
        add_26: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_12, 1e-06);  mean_12 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:217 in _norm, code: return hidden_states * torch.pow(mean_squared, -0.5)
        pow_26: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_26, -0.5);  add_26 = None
        mul_45: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_64, pow_26);  convert_element_type_64 = pow_26 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:222 in forward, code: normed_output = normed_output * self.weight.float()
        convert_element_type_65: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg27_1, torch.float32);  arg27_1 = None
        mul_46: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_45, convert_element_type_65);  mul_45 = convert_element_type_65 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:223 in forward, code: return normed_output.type_as(hidden_states)
        convert_element_type_66: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_46, torch.bfloat16);  mul_46 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1075 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_36: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_66, [2048, 5376])
        permute_19: "bf16[5376, 21504][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        mm_11: "bf16[2048, 21504][21504, 1]cuda:0" = torch.ops.aten.mm.default(view_36, permute_19);  view_36 = permute_19 = None
        view_37: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.reshape.default(mm_11, [1, 2048, 21504]);  mm_11 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/activations.py:51 in forward, code: return self.act(input)
        convert_element_type_69: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_37, torch.float32);  view_37 = None
        mul_51: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_69, 0.5)
        mul_47: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_69, convert_element_type_69)
        mul_48: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_47, convert_element_type_69);  mul_47 = None
        mul_49: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_48, 0.044715);  mul_48 = None
        add_27: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_69, mul_49);  convert_element_type_69 = mul_49 = None
        mul_50: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_27, 0.7978845608028654);  add_27 = None
        tanh_1: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.tanh.default(mul_50);  mul_50 = None
        add_28: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_1, 1);  tanh_1 = None
        mul_52: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_51, add_28);  mul_51 = add_28 = None
        convert_element_type_70: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_52, torch.bfloat16);  mul_52 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1075 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_38: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_66, [2048, 5376]);  convert_element_type_66 = None
        permute_20: "bf16[5376, 21504][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        mm_12: "bf16[2048, 21504][21504, 1]cuda:0" = torch.ops.aten.mm.default(view_38, permute_20);  view_38 = permute_20 = None
        view_39: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.reshape.default(mm_12, [1, 2048, 21504]);  mm_12 = None
        mul_53: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_70, view_39);  convert_element_type_70 = view_39 = None
        view_40: "bf16[2048, 21504][21504, 1]cuda:0" = torch.ops.aten.reshape.default(mul_53, [2048, 21504]);  mul_53 = None
        permute_21: "bf16[21504, 5376][1, 21504]cuda:0" = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
        mm_13: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.mm.default(view_40, permute_21);  view_40 = permute_21 = None
        view_41: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.reshape.default(mm_13, [1, 2048, 5376]);  mm_13 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:220 in forward, code: normed_output = self._norm(hidden_states.float())
        convert_element_type_75: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_41, torch.float32);  view_41 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:215 in _norm, code: mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        pow_27: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_75, 2)
        mean_13: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_27, [-1], True);  pow_27 = None
        add_29: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_13, 1e-06);  mean_13 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:217 in _norm, code: return hidden_states * torch.pow(mean_squared, -0.5)
        pow_28: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_29, -0.5);  add_29 = None
        mul_54: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_75, pow_28);  convert_element_type_75 = pow_28 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:222 in forward, code: normed_output = normed_output * self.weight.float()
        convert_element_type_76: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg31_1, torch.float32);  arg31_1 = None
        mul_55: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_54, convert_element_type_76);  mul_54 = convert_element_type_76 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:223 in forward, code: return normed_output.type_as(hidden_states)
        convert_element_type_77: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_55, torch.bfloat16);  mul_55 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1418 in forward, code: hidden_states = residual + hidden_states
        add_30: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.add.Tensor(add_25, convert_element_type_77);  add_25 = convert_element_type_77 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1429 in forward, code: hidden_states *= self.layer_scalar
        mul_56: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_30, arg32_1);  add_30 = arg32_1 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:220 in forward, code: normed_output = self._norm(hidden_states.float())
        convert_element_type_78: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_56, torch.float32)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:215 in _norm, code: mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        pow_29: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_78, 2)
        mean_14: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_29, [-1], True);  pow_29 = None
        add_31: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_14, 1e-06);  mean_14 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:217 in _norm, code: return hidden_states * torch.pow(mean_squared, -0.5)
        pow_30: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_31, -0.5);  add_31 = None
        mul_57: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_78, pow_30);  convert_element_type_78 = pow_30 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:222 in forward, code: normed_output = normed_output * self.weight.float()
        convert_element_type_79: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg33_1, torch.float32);  arg33_1 = None
        mul_58: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_57, convert_element_type_79);  mul_57 = convert_element_type_79 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:223 in forward, code: return normed_output.type_as(hidden_states)
        convert_element_type_80: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_58, torch.bfloat16);  mul_58 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1217 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape)
        view_42: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_80, [2048, 5376])
        permute_22: "bf16[5376, 8192][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        mm_14: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_42, permute_22);  view_42 = permute_22 = None
        view_43: "bf16[1, 2048, 8192][16777216, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_14, [1, 2048, 8192]);  mm_14 = None
        view_44: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_43, [1, 2048, -1, 256]);  view_43 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:220 in forward, code: normed_output = self._norm(hidden_states.float())
        convert_element_type_83: "f32[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_44, torch.float32);  view_44 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:215 in _norm, code: mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        pow_31: "f32[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_83, 2)
        mean_15: "f32[1, 2048, 32, 1][65536, 32, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_31, [-1], True);  pow_31 = None
        add_32: "f32[1, 2048, 32, 1][65536, 32, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_15, 1e-06);  mean_15 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:217 in _norm, code: return hidden_states * torch.pow(mean_squared, -0.5)
        pow_32: "f32[1, 2048, 32, 1][65536, 32, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_32, -0.5);  add_32 = None
        mul_59: "f32[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_83, pow_32);  convert_element_type_83 = pow_32 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:222 in forward, code: normed_output = normed_output * self.weight.float()
        convert_element_type_84: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg35_1, torch.float32);  arg35_1 = None
        mul_60: "f32[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_59, convert_element_type_84);  mul_59 = convert_element_type_84 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:223 in forward, code: return normed_output.type_as(hidden_states)
        convert_element_type_85: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_60, torch.bfloat16);  mul_60 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:796 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_35: "bf16[1, 2048, 1, 256][524288, 256, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_4, 2)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:798 in apply_rotary_pos_emb, code: return (x * cos) + (rotate_half(x) * sin)
        mul_61: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_85, unsqueeze_35);  unsqueeze_35 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:775 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_16: "bf16[1, 2048, 32, 128][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_85, 3, 128, 9223372036854775807)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:776 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_4: "bf16[1, 2048, 32, 128][8388608, 4096, 128, 1]cuda:0" = torch.ops.aten.neg.default(slice_16);  slice_16 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:774 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_15: "bf16[1, 2048, 32, 128][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_85, 3, 0, 128);  convert_element_type_85 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:776 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_6: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_4, slice_15], -1);  neg_4 = slice_15 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:797 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_36: "bf16[1, 2048, 1, 256][524288, 256, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_5, 2)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:798 in apply_rotary_pos_emb, code: return (x * cos) + (rotate_half(x) * sin)
        mul_62: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_6, unsqueeze_36);  cat_6 = unsqueeze_36 = None
        add_33: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_61, mul_62);  mul_61 = mul_62 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1220 in forward, code: query_states = query_states.transpose(1, 2)
        permute_23: "bf16[1, 32, 2048, 256][16777216, 256, 8192, 1]cuda:0" = torch.ops.aten.permute.default(add_33, [0, 2, 1, 3]);  add_33 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1231 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape)
        view_45: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_80, [2048, 5376])
        permute_24: "bf16[5376, 4096][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
        mm_15: "bf16[2048, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_45, permute_24);  view_45 = permute_24 = None
        view_46: "bf16[1, 2048, 4096][8388608, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_15, [1, 2048, 4096]);  mm_15 = None
        view_47: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_46, [1, 2048, -1, 256]);  view_46 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:220 in forward, code: normed_output = self._norm(hidden_states.float())
        convert_element_type_90: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_47, torch.float32);  view_47 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:215 in _norm, code: mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        pow_33: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_90, 2)
        mean_16: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_33, [-1], True);  pow_33 = None
        add_34: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_16, 1e-06);  mean_16 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:217 in _norm, code: return hidden_states * torch.pow(mean_squared, -0.5)
        pow_34: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_34, -0.5);  add_34 = None
        mul_63: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_90, pow_34);  convert_element_type_90 = pow_34 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:222 in forward, code: normed_output = normed_output * self.weight.float()
        convert_element_type_91: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg38_1, torch.float32);  arg38_1 = None
        mul_64: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_63, convert_element_type_91);  mul_63 = convert_element_type_91 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:223 in forward, code: return normed_output.type_as(hidden_states)
        convert_element_type_92: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_64, torch.bfloat16);  mul_64 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:796 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_37: "bf16[1, 2048, 1, 256][524288, 256, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_4, 2)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:798 in apply_rotary_pos_emb, code: return (x * cos) + (rotate_half(x) * sin)
        mul_65: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_92, unsqueeze_37);  unsqueeze_37 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:775 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_18: "bf16[1, 2048, 16, 128][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_92, 3, 128, 9223372036854775807)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:776 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_5: "bf16[1, 2048, 16, 128][4194304, 2048, 128, 1]cuda:0" = torch.ops.aten.neg.default(slice_18);  slice_18 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:774 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_17: "bf16[1, 2048, 16, 128][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_92, 3, 0, 128);  convert_element_type_92 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:776 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_7: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_5, slice_17], -1);  neg_5 = slice_17 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:797 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_38: "bf16[1, 2048, 1, 256][524288, 256, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_5, 2)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:798 in apply_rotary_pos_emb, code: return (x * cos) + (rotate_half(x) * sin)
        mul_66: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_7, unsqueeze_38);  cat_7 = unsqueeze_38 = None
        add_35: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_65, mul_66);  mul_65 = mul_66 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1236 in forward, code: key_states = key_states.transpose(1, 2)
        permute_26: "bf16[1, 16, 2048, 256][8388608, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(add_35, [0, 2, 1, 3]);  add_35 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/integrations/sdpa_attention.py:26 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_39: "bf16[1, 16, 1, 2048, 256][8388608, 256, 8388608, 4096, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_26, 2);  permute_26 = None
        expand_10: "bf16[1, 16, 2, 2048, 256][8388608, 256, 0, 4096, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_39, [1, 16, 2, 2048, 256]);  unsqueeze_39 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/integrations/sdpa_attention.py:27 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_8: "bf16[1, 16, 2, 2048, 256][16777216, 1048576, 524288, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_51: "bf16[1, 32, 2048, 256][16777216, 524288, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_8, [1, 32, 2048, 256]);  clone_8 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1232 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states
        view_48: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_80, [2048, 5376]);  convert_element_type_80 = None
        permute_25: "bf16[5376, 4096][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        mm_16: "bf16[2048, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_48, permute_25);  view_48 = permute_25 = None
        view_49: "bf16[1, 2048, 4096][8388608, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_16, [1, 2048, 4096]);  mm_16 = None
        view_50: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_49, [1, 2048, -1, 256]);  view_49 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:220 in forward, code: normed_output = self._norm(hidden_states.float())
        convert_element_type_93: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_50, torch.float32);  view_50 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:215 in _norm, code: mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        pow_35: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_93, 2)
        mean_17: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_35, [-1], True);  pow_35 = None
        add_36: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_17, 1e-06);  mean_17 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:217 in _norm, code: return hidden_states * torch.pow(mean_squared, -0.5)
        pow_36: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_36, -0.5);  add_36 = None
        mul_67: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_93, pow_36);  convert_element_type_93 = pow_36 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:223 in forward, code: return normed_output.type_as(hidden_states)
        convert_element_type_94: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_67, torch.bfloat16);  mul_67 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1239 in forward, code: value_states = value_states.transpose(1, 2)
        permute_27: "bf16[1, 16, 2048, 256][8388608, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_94, [0, 2, 1, 3]);  convert_element_type_94 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/integrations/sdpa_attention.py:26 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_40: "bf16[1, 16, 1, 2048, 256][8388608, 256, 8388608, 4096, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_27, 2);  permute_27 = None
        expand_11: "bf16[1, 16, 2, 2048, 256][8388608, 256, 0, 4096, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_40, [1, 16, 2, 2048, 256]);  unsqueeze_40 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/integrations/sdpa_attention.py:27 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_9: "bf16[1, 16, 2, 2048, 256][16777216, 1048576, 524288, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_52: "bf16[1, 32, 2048, 256][16777216, 524288, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_9, [1, 32, 2048, 256]);  clone_9 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/integrations/sdpa_attention.py:174 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_8: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_7: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "bf16[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.where.self(expand_1, full_default_8, full_default_7);  full_default_8 = full_default_7 = None
        expand_12: "bf16[1, 32, 2048, 2048][4194304, 0, 2048, 1]cuda:0" = torch.ops.aten.expand.default(where_2, [1, 32, 2048, 2048]);  where_2 = None
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_23, view_51, view_52, expand_12, False, scale = 1.0);  permute_23 = view_51 = view_52 = expand_12 = None
        getitem_8: "bf16[1, 32, 2048, 256][16777216, 256, 8192, 1]cuda:0" = _scaled_dot_product_efficient_attention_2[0];  _scaled_dot_product_efficient_attention_2 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/integrations/sdpa_attention.py:184 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_28: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.permute.default(getitem_8, [0, 2, 1, 3]);  getitem_8 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1262 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_53: "bf16[1, 2048, 8192][16777216, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(permute_28, [1, 2048, -1]);  permute_28 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1263 in forward, code: attn_output = self.o_proj(attn_output)
        view_54: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(view_53, [2048, 8192]);  view_53 = None
        permute_29: "bf16[8192, 5376][1, 8192]cuda:0" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        mm_17: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.mm.default(view_54, permute_29);  view_54 = permute_29 = None
        view_55: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.reshape.default(mm_17, [1, 2048, 5376]);  mm_17 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:220 in forward, code: normed_output = self._norm(hidden_states.float())
        convert_element_type_97: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_55, torch.float32);  view_55 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:215 in _norm, code: mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        pow_37: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_97, 2)
        mean_18: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_37, [-1], True);  pow_37 = None
        add_37: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_18, 1e-06);  mean_18 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:217 in _norm, code: return hidden_states * torch.pow(mean_squared, -0.5)
        pow_38: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_37, -0.5);  add_37 = None
        mul_68: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_97, pow_38);  convert_element_type_97 = pow_38 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:222 in forward, code: normed_output = normed_output * self.weight.float()
        convert_element_type_98: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg40_1, torch.float32);  arg40_1 = None
        mul_69: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_68, convert_element_type_98);  mul_68 = convert_element_type_98 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:223 in forward, code: return normed_output.type_as(hidden_states)
        convert_element_type_99: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_69, torch.bfloat16);  mul_69 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1397 in forward, code: hidden_states = residual + hidden_states
        add_38: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_56, convert_element_type_99);  mul_56 = convert_element_type_99 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:220 in forward, code: normed_output = self._norm(hidden_states.float())
        convert_element_type_100: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_38, torch.float32)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:215 in _norm, code: mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        pow_39: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_100, 2)
        mean_19: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_39, [-1], True);  pow_39 = None
        add_39: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_19, 1e-06);  mean_19 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:217 in _norm, code: return hidden_states * torch.pow(mean_squared, -0.5)
        pow_40: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_39, -0.5);  add_39 = None
        mul_70: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_100, pow_40);  convert_element_type_100 = pow_40 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:222 in forward, code: normed_output = normed_output * self.weight.float()
        convert_element_type_101: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg41_1, torch.float32);  arg41_1 = None
        mul_71: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_70, convert_element_type_101);  mul_70 = convert_element_type_101 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:223 in forward, code: return normed_output.type_as(hidden_states)
        convert_element_type_102: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_71, torch.bfloat16);  mul_71 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1075 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_56: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_102, [2048, 5376])
        permute_30: "bf16[5376, 21504][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        mm_18: "bf16[2048, 21504][21504, 1]cuda:0" = torch.ops.aten.mm.default(view_56, permute_30);  view_56 = permute_30 = None
        view_57: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.reshape.default(mm_18, [1, 2048, 21504]);  mm_18 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/activations.py:51 in forward, code: return self.act(input)
        convert_element_type_105: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_57, torch.float32);  view_57 = None
        mul_76: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_105, 0.5)
        mul_72: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_105, convert_element_type_105)
        mul_73: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_72, convert_element_type_105);  mul_72 = None
        mul_74: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_73, 0.044715);  mul_73 = None
        add_40: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_105, mul_74);  convert_element_type_105 = mul_74 = None
        mul_75: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_40, 0.7978845608028654);  add_40 = None
        tanh_2: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.tanh.default(mul_75);  mul_75 = None
        add_41: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_2, 1);  tanh_2 = None
        mul_77: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_76, add_41);  mul_76 = add_41 = None
        convert_element_type_106: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_77, torch.bfloat16);  mul_77 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1075 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_58: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_102, [2048, 5376]);  convert_element_type_102 = None
        permute_31: "bf16[5376, 21504][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        mm_19: "bf16[2048, 21504][21504, 1]cuda:0" = torch.ops.aten.mm.default(view_58, permute_31);  view_58 = permute_31 = None
        view_59: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.reshape.default(mm_19, [1, 2048, 21504]);  mm_19 = None
        mul_78: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_106, view_59);  convert_element_type_106 = view_59 = None
        view_60: "bf16[2048, 21504][21504, 1]cuda:0" = torch.ops.aten.reshape.default(mul_78, [2048, 21504]);  mul_78 = None
        permute_32: "bf16[21504, 5376][1, 21504]cuda:0" = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
        mm_20: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.mm.default(view_60, permute_32);  view_60 = permute_32 = None
        view_61: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.reshape.default(mm_20, [1, 2048, 5376]);  mm_20 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:220 in forward, code: normed_output = self._norm(hidden_states.float())
        convert_element_type_111: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_61, torch.float32);  view_61 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:215 in _norm, code: mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        pow_41: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_111, 2)
        mean_20: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_41, [-1], True);  pow_41 = None
        add_42: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_20, 1e-06);  mean_20 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:217 in _norm, code: return hidden_states * torch.pow(mean_squared, -0.5)
        pow_42: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_42, -0.5);  add_42 = None
        mul_79: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_111, pow_42);  convert_element_type_111 = pow_42 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:222 in forward, code: normed_output = normed_output * self.weight.float()
        convert_element_type_112: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg45_1, torch.float32);  arg45_1 = None
        mul_80: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_79, convert_element_type_112);  mul_79 = convert_element_type_112 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:223 in forward, code: return normed_output.type_as(hidden_states)
        convert_element_type_113: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_80, torch.bfloat16);  mul_80 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1418 in forward, code: hidden_states = residual + hidden_states
        add_43: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.add.Tensor(add_38, convert_element_type_113);  add_38 = convert_element_type_113 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1429 in forward, code: hidden_states *= self.layer_scalar
        mul_81: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_43, arg46_1);  add_43 = arg46_1 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:220 in forward, code: normed_output = self._norm(hidden_states.float())
        convert_element_type_114: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_81, torch.float32)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:215 in _norm, code: mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        pow_43: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_114, 2)
        mean_21: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_43, [-1], True);  pow_43 = None
        add_44: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_21, 1e-06);  mean_21 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:217 in _norm, code: return hidden_states * torch.pow(mean_squared, -0.5)
        pow_44: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_44, -0.5);  add_44 = None
        mul_82: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_114, pow_44);  convert_element_type_114 = pow_44 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:222 in forward, code: normed_output = normed_output * self.weight.float()
        convert_element_type_115: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg47_1, torch.float32);  arg47_1 = None
        mul_83: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_82, convert_element_type_115);  mul_82 = convert_element_type_115 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:223 in forward, code: return normed_output.type_as(hidden_states)
        convert_element_type_116: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_83, torch.bfloat16);  mul_83 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1217 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape)
        view_62: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_116, [2048, 5376])
        permute_33: "bf16[5376, 8192][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
        mm_21: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_62, permute_33);  view_62 = permute_33 = None
        view_63: "bf16[1, 2048, 8192][16777216, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_21, [1, 2048, 8192]);  mm_21 = None
        view_64: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_63, [1, 2048, -1, 256]);  view_63 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:220 in forward, code: normed_output = self._norm(hidden_states.float())
        convert_element_type_119: "f32[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_64, torch.float32);  view_64 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:215 in _norm, code: mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        pow_45: "f32[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_119, 2)
        mean_22: "f32[1, 2048, 32, 1][65536, 32, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_45, [-1], True);  pow_45 = None
        add_45: "f32[1, 2048, 32, 1][65536, 32, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_22, 1e-06);  mean_22 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:217 in _norm, code: return hidden_states * torch.pow(mean_squared, -0.5)
        pow_46: "f32[1, 2048, 32, 1][65536, 32, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_45, -0.5);  add_45 = None
        mul_84: "f32[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_119, pow_46);  convert_element_type_119 = pow_46 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:222 in forward, code: normed_output = normed_output * self.weight.float()
        convert_element_type_120: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg49_1, torch.float32);  arg49_1 = None
        mul_85: "f32[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_84, convert_element_type_120);  mul_84 = convert_element_type_120 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:223 in forward, code: return normed_output.type_as(hidden_states)
        convert_element_type_121: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_85, torch.bfloat16);  mul_85 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:796 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_41: "bf16[1, 2048, 1, 256][524288, 256, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_4, 2)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:798 in apply_rotary_pos_emb, code: return (x * cos) + (rotate_half(x) * sin)
        mul_86: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_121, unsqueeze_41);  unsqueeze_41 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:775 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_20: "bf16[1, 2048, 32, 128][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_121, 3, 128, 9223372036854775807)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:776 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_6: "bf16[1, 2048, 32, 128][8388608, 4096, 128, 1]cuda:0" = torch.ops.aten.neg.default(slice_20);  slice_20 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:774 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_19: "bf16[1, 2048, 32, 128][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_121, 3, 0, 128);  convert_element_type_121 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:776 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_8: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_6, slice_19], -1);  neg_6 = slice_19 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:797 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_42: "bf16[1, 2048, 1, 256][524288, 256, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_5, 2)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:798 in apply_rotary_pos_emb, code: return (x * cos) + (rotate_half(x) * sin)
        mul_87: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_8, unsqueeze_42);  cat_8 = unsqueeze_42 = None
        add_46: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_86, mul_87);  mul_86 = mul_87 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1220 in forward, code: query_states = query_states.transpose(1, 2)
        permute_34: "bf16[1, 32, 2048, 256][16777216, 256, 8192, 1]cuda:0" = torch.ops.aten.permute.default(add_46, [0, 2, 1, 3]);  add_46 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1231 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape)
        view_65: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_116, [2048, 5376])
        permute_35: "bf16[5376, 4096][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        mm_22: "bf16[2048, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_65, permute_35);  view_65 = permute_35 = None
        view_66: "bf16[1, 2048, 4096][8388608, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_22, [1, 2048, 4096]);  mm_22 = None
        view_67: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_66, [1, 2048, -1, 256]);  view_66 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:220 in forward, code: normed_output = self._norm(hidden_states.float())
        convert_element_type_126: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_67, torch.float32);  view_67 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:215 in _norm, code: mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        pow_47: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_126, 2)
        mean_23: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_47, [-1], True);  pow_47 = None
        add_47: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_23, 1e-06);  mean_23 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:217 in _norm, code: return hidden_states * torch.pow(mean_squared, -0.5)
        pow_48: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_47, -0.5);  add_47 = None
        mul_88: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_126, pow_48);  convert_element_type_126 = pow_48 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:222 in forward, code: normed_output = normed_output * self.weight.float()
        convert_element_type_127: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg52_1, torch.float32);  arg52_1 = None
        mul_89: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_88, convert_element_type_127);  mul_88 = convert_element_type_127 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:223 in forward, code: return normed_output.type_as(hidden_states)
        convert_element_type_128: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_89, torch.bfloat16);  mul_89 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:796 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_43: "bf16[1, 2048, 1, 256][524288, 256, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_4, 2)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:798 in apply_rotary_pos_emb, code: return (x * cos) + (rotate_half(x) * sin)
        mul_90: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_128, unsqueeze_43);  unsqueeze_43 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:775 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_22: "bf16[1, 2048, 16, 128][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_128, 3, 128, 9223372036854775807)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:776 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_7: "bf16[1, 2048, 16, 128][4194304, 2048, 128, 1]cuda:0" = torch.ops.aten.neg.default(slice_22);  slice_22 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:774 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_21: "bf16[1, 2048, 16, 128][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_128, 3, 0, 128);  convert_element_type_128 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:776 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_9: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_7, slice_21], -1);  neg_7 = slice_21 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:797 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_44: "bf16[1, 2048, 1, 256][524288, 256, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_5, 2)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:798 in apply_rotary_pos_emb, code: return (x * cos) + (rotate_half(x) * sin)
        mul_91: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_9, unsqueeze_44);  cat_9 = unsqueeze_44 = None
        add_48: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_90, mul_91);  mul_90 = mul_91 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1236 in forward, code: key_states = key_states.transpose(1, 2)
        permute_37: "bf16[1, 16, 2048, 256][8388608, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(add_48, [0, 2, 1, 3]);  add_48 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/integrations/sdpa_attention.py:26 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_45: "bf16[1, 16, 1, 2048, 256][8388608, 256, 8388608, 4096, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_37, 2);  permute_37 = None
        expand_13: "bf16[1, 16, 2, 2048, 256][8388608, 256, 0, 4096, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_45, [1, 16, 2, 2048, 256]);  unsqueeze_45 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/integrations/sdpa_attention.py:27 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_10: "bf16[1, 16, 2, 2048, 256][16777216, 1048576, 524288, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_71: "bf16[1, 32, 2048, 256][16777216, 524288, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_10, [1, 32, 2048, 256]);  clone_10 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1232 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states
        view_68: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_116, [2048, 5376]);  convert_element_type_116 = None
        permute_36: "bf16[5376, 4096][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        mm_23: "bf16[2048, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_68, permute_36);  view_68 = permute_36 = None
        view_69: "bf16[1, 2048, 4096][8388608, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_23, [1, 2048, 4096]);  mm_23 = None
        view_70: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_69, [1, 2048, -1, 256]);  view_69 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:220 in forward, code: normed_output = self._norm(hidden_states.float())
        convert_element_type_129: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_70, torch.float32);  view_70 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:215 in _norm, code: mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        pow_49: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_129, 2)
        mean_24: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_49, [-1], True);  pow_49 = None
        add_49: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_24, 1e-06);  mean_24 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:217 in _norm, code: return hidden_states * torch.pow(mean_squared, -0.5)
        pow_50: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_49, -0.5);  add_49 = None
        mul_92: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_129, pow_50);  convert_element_type_129 = pow_50 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:223 in forward, code: return normed_output.type_as(hidden_states)
        convert_element_type_130: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_92, torch.bfloat16);  mul_92 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1239 in forward, code: value_states = value_states.transpose(1, 2)
        permute_38: "bf16[1, 16, 2048, 256][8388608, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_130, [0, 2, 1, 3]);  convert_element_type_130 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/integrations/sdpa_attention.py:26 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_46: "bf16[1, 16, 1, 2048, 256][8388608, 256, 8388608, 4096, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_38, 2);  permute_38 = None
        expand_14: "bf16[1, 16, 2, 2048, 256][8388608, 256, 0, 4096, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_46, [1, 16, 2, 2048, 256]);  unsqueeze_46 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/integrations/sdpa_attention.py:27 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_11: "bf16[1, 16, 2, 2048, 256][16777216, 1048576, 524288, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
        view_72: "bf16[1, 32, 2048, 256][16777216, 524288, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_11, [1, 32, 2048, 256]);  clone_11 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/integrations/sdpa_attention.py:174 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_10: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_9: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "bf16[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.where.self(expand_1, full_default_10, full_default_9);  full_default_10 = full_default_9 = None
        expand_15: "bf16[1, 32, 2048, 2048][4194304, 0, 2048, 1]cuda:0" = torch.ops.aten.expand.default(where_3, [1, 32, 2048, 2048]);  where_3 = None
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_34, view_71, view_72, expand_15, False, scale = 1.0);  permute_34 = view_71 = view_72 = expand_15 = None
        getitem_12: "bf16[1, 32, 2048, 256][16777216, 256, 8192, 1]cuda:0" = _scaled_dot_product_efficient_attention_3[0];  _scaled_dot_product_efficient_attention_3 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/integrations/sdpa_attention.py:184 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_39: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.permute.default(getitem_12, [0, 2, 1, 3]);  getitem_12 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1262 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_73: "bf16[1, 2048, 8192][16777216, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(permute_39, [1, 2048, -1]);  permute_39 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1263 in forward, code: attn_output = self.o_proj(attn_output)
        view_74: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(view_73, [2048, 8192]);  view_73 = None
        permute_40: "bf16[8192, 5376][1, 8192]cuda:0" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        mm_24: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.mm.default(view_74, permute_40);  view_74 = permute_40 = None
        view_75: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.reshape.default(mm_24, [1, 2048, 5376]);  mm_24 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:220 in forward, code: normed_output = self._norm(hidden_states.float())
        convert_element_type_133: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_75, torch.float32);  view_75 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:215 in _norm, code: mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        pow_51: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_133, 2)
        mean_25: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_51, [-1], True);  pow_51 = None
        add_50: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_25, 1e-06);  mean_25 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:217 in _norm, code: return hidden_states * torch.pow(mean_squared, -0.5)
        pow_52: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_50, -0.5);  add_50 = None
        mul_93: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_133, pow_52);  convert_element_type_133 = pow_52 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:222 in forward, code: normed_output = normed_output * self.weight.float()
        convert_element_type_134: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg54_1, torch.float32);  arg54_1 = None
        mul_94: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_93, convert_element_type_134);  mul_93 = convert_element_type_134 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:223 in forward, code: return normed_output.type_as(hidden_states)
        convert_element_type_135: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_94, torch.bfloat16);  mul_94 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1397 in forward, code: hidden_states = residual + hidden_states
        add_51: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_81, convert_element_type_135);  mul_81 = convert_element_type_135 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:220 in forward, code: normed_output = self._norm(hidden_states.float())
        convert_element_type_136: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_51, torch.float32)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:215 in _norm, code: mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        pow_53: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_136, 2)
        mean_26: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_53, [-1], True);  pow_53 = None
        add_52: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_26, 1e-06);  mean_26 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:217 in _norm, code: return hidden_states * torch.pow(mean_squared, -0.5)
        pow_54: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_52, -0.5);  add_52 = None
        mul_95: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_136, pow_54);  convert_element_type_136 = pow_54 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:222 in forward, code: normed_output = normed_output * self.weight.float()
        convert_element_type_137: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg55_1, torch.float32);  arg55_1 = None
        mul_96: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_95, convert_element_type_137);  mul_95 = convert_element_type_137 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:223 in forward, code: return normed_output.type_as(hidden_states)
        convert_element_type_138: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_96, torch.bfloat16);  mul_96 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1075 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_76: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_138, [2048, 5376])
        permute_41: "bf16[5376, 21504][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        mm_25: "bf16[2048, 21504][21504, 1]cuda:0" = torch.ops.aten.mm.default(view_76, permute_41);  view_76 = permute_41 = None
        view_77: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.reshape.default(mm_25, [1, 2048, 21504]);  mm_25 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/activations.py:51 in forward, code: return self.act(input)
        convert_element_type_141: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_77, torch.float32);  view_77 = None
        mul_101: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_141, 0.5)
        mul_97: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_141, convert_element_type_141)
        mul_98: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_97, convert_element_type_141);  mul_97 = None
        mul_99: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_98, 0.044715);  mul_98 = None
        add_53: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_141, mul_99);  convert_element_type_141 = mul_99 = None
        mul_100: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_53, 0.7978845608028654);  add_53 = None
        tanh_3: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.tanh.default(mul_100);  mul_100 = None
        add_54: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_3, 1);  tanh_3 = None
        mul_102: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_101, add_54);  mul_101 = add_54 = None
        convert_element_type_142: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_102, torch.bfloat16);  mul_102 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1075 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_78: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_138, [2048, 5376]);  convert_element_type_138 = None
        permute_42: "bf16[5376, 21504][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        mm_26: "bf16[2048, 21504][21504, 1]cuda:0" = torch.ops.aten.mm.default(view_78, permute_42);  view_78 = permute_42 = None
        view_79: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.reshape.default(mm_26, [1, 2048, 21504]);  mm_26 = None
        mul_103: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_142, view_79);  convert_element_type_142 = view_79 = None
        view_80: "bf16[2048, 21504][21504, 1]cuda:0" = torch.ops.aten.reshape.default(mul_103, [2048, 21504]);  mul_103 = None
        permute_43: "bf16[21504, 5376][1, 21504]cuda:0" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        mm_27: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.mm.default(view_80, permute_43);  view_80 = permute_43 = None
        view_81: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.reshape.default(mm_27, [1, 2048, 5376]);  mm_27 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:220 in forward, code: normed_output = self._norm(hidden_states.float())
        convert_element_type_147: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_81, torch.float32);  view_81 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:215 in _norm, code: mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        pow_55: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_147, 2)
        mean_27: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_55, [-1], True);  pow_55 = None
        add_55: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_27, 1e-06);  mean_27 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:217 in _norm, code: return hidden_states * torch.pow(mean_squared, -0.5)
        pow_56: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_55, -0.5);  add_55 = None
        mul_104: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_147, pow_56);  convert_element_type_147 = pow_56 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:222 in forward, code: normed_output = normed_output * self.weight.float()
        convert_element_type_148: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg59_1, torch.float32);  arg59_1 = None
        mul_105: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_104, convert_element_type_148);  mul_104 = convert_element_type_148 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:223 in forward, code: return normed_output.type_as(hidden_states)
        convert_element_type_149: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_105, torch.bfloat16);  mul_105 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1418 in forward, code: hidden_states = residual + hidden_states
        add_56: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.add.Tensor(add_51, convert_element_type_149);  add_51 = convert_element_type_149 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1429 in forward, code: hidden_states *= self.layer_scalar
        mul_106: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_56, arg60_1);  add_56 = arg60_1 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:220 in forward, code: normed_output = self._norm(hidden_states.float())
        convert_element_type_150: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_106, torch.float32)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:215 in _norm, code: mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        pow_57: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_150, 2)
        mean_28: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_57, [-1], True);  pow_57 = None
        add_57: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_28, 1e-06);  mean_28 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:217 in _norm, code: return hidden_states * torch.pow(mean_squared, -0.5)
        pow_58: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_57, -0.5);  add_57 = None
        mul_107: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_150, pow_58);  convert_element_type_150 = pow_58 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:222 in forward, code: normed_output = normed_output * self.weight.float()
        convert_element_type_151: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg61_1, torch.float32);  arg61_1 = None
        mul_108: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_107, convert_element_type_151);  mul_107 = convert_element_type_151 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:223 in forward, code: return normed_output.type_as(hidden_states)
        convert_element_type_152: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_108, torch.bfloat16);  mul_108 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1217 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape)
        view_82: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_152, [2048, 5376])
        permute_44: "bf16[5376, 8192][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
        mm_28: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_82, permute_44);  view_82 = permute_44 = None
        view_83: "bf16[1, 2048, 8192][16777216, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_28, [1, 2048, 8192]);  mm_28 = None
        view_84: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_83, [1, 2048, -1, 256]);  view_83 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:220 in forward, code: normed_output = self._norm(hidden_states.float())
        convert_element_type_155: "f32[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_84, torch.float32);  view_84 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:215 in _norm, code: mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        pow_59: "f32[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_155, 2)
        mean_29: "f32[1, 2048, 32, 1][65536, 32, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_59, [-1], True);  pow_59 = None
        add_58: "f32[1, 2048, 32, 1][65536, 32, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_29, 1e-06);  mean_29 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:217 in _norm, code: return hidden_states * torch.pow(mean_squared, -0.5)
        pow_60: "f32[1, 2048, 32, 1][65536, 32, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_58, -0.5);  add_58 = None
        mul_109: "f32[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_155, pow_60);  convert_element_type_155 = pow_60 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:222 in forward, code: normed_output = normed_output * self.weight.float()
        convert_element_type_156: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg63_1, torch.float32);  arg63_1 = None
        mul_110: "f32[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_109, convert_element_type_156);  mul_109 = convert_element_type_156 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:223 in forward, code: return normed_output.type_as(hidden_states)
        convert_element_type_157: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_110, torch.bfloat16);  mul_110 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:796 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_47: "bf16[1, 2048, 1, 256][524288, 256, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_4, 2)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:798 in apply_rotary_pos_emb, code: return (x * cos) + (rotate_half(x) * sin)
        mul_111: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_157, unsqueeze_47);  unsqueeze_47 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:775 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_24: "bf16[1, 2048, 32, 128][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_157, 3, 128, 9223372036854775807)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:776 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_8: "bf16[1, 2048, 32, 128][8388608, 4096, 128, 1]cuda:0" = torch.ops.aten.neg.default(slice_24);  slice_24 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:774 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_23: "bf16[1, 2048, 32, 128][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_157, 3, 0, 128);  convert_element_type_157 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:776 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_10: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_8, slice_23], -1);  neg_8 = slice_23 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:797 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_48: "bf16[1, 2048, 1, 256][524288, 256, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_5, 2)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:798 in apply_rotary_pos_emb, code: return (x * cos) + (rotate_half(x) * sin)
        mul_112: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_10, unsqueeze_48);  cat_10 = unsqueeze_48 = None
        add_59: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_111, mul_112);  mul_111 = mul_112 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1220 in forward, code: query_states = query_states.transpose(1, 2)
        permute_45: "bf16[1, 32, 2048, 256][16777216, 256, 8192, 1]cuda:0" = torch.ops.aten.permute.default(add_59, [0, 2, 1, 3]);  add_59 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1231 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape)
        view_85: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_152, [2048, 5376])
        permute_46: "bf16[5376, 4096][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        mm_29: "bf16[2048, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_85, permute_46);  view_85 = permute_46 = None
        view_86: "bf16[1, 2048, 4096][8388608, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_29, [1, 2048, 4096]);  mm_29 = None
        view_87: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_86, [1, 2048, -1, 256]);  view_86 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:220 in forward, code: normed_output = self._norm(hidden_states.float())
        convert_element_type_162: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_87, torch.float32);  view_87 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:215 in _norm, code: mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        pow_61: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_162, 2)
        mean_30: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_61, [-1], True);  pow_61 = None
        add_60: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_30, 1e-06);  mean_30 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:217 in _norm, code: return hidden_states * torch.pow(mean_squared, -0.5)
        pow_62: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_60, -0.5);  add_60 = None
        mul_113: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_162, pow_62);  convert_element_type_162 = pow_62 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:222 in forward, code: normed_output = normed_output * self.weight.float()
        convert_element_type_163: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg66_1, torch.float32);  arg66_1 = None
        mul_114: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_113, convert_element_type_163);  mul_113 = convert_element_type_163 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:223 in forward, code: return normed_output.type_as(hidden_states)
        convert_element_type_164: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_114, torch.bfloat16);  mul_114 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:796 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_49: "bf16[1, 2048, 1, 256][524288, 256, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_4, 2);  convert_element_type_4 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:798 in apply_rotary_pos_emb, code: return (x * cos) + (rotate_half(x) * sin)
        mul_115: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_164, unsqueeze_49);  unsqueeze_49 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:775 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_26: "bf16[1, 2048, 16, 128][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_164, 3, 128, 9223372036854775807)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:776 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_9: "bf16[1, 2048, 16, 128][4194304, 2048, 128, 1]cuda:0" = torch.ops.aten.neg.default(slice_26);  slice_26 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:774 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_25: "bf16[1, 2048, 16, 128][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_164, 3, 0, 128);  convert_element_type_164 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:776 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_11: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.cat.default([neg_9, slice_25], -1);  neg_9 = slice_25 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:797 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_50: "bf16[1, 2048, 1, 256][524288, 256, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_5, 2);  convert_element_type_5 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:798 in apply_rotary_pos_emb, code: return (x * cos) + (rotate_half(x) * sin)
        mul_116: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_11, unsqueeze_50);  cat_11 = unsqueeze_50 = None
        add_61: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_115, mul_116);  mul_115 = mul_116 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1236 in forward, code: key_states = key_states.transpose(1, 2)
        permute_48: "bf16[1, 16, 2048, 256][8388608, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(add_61, [0, 2, 1, 3]);  add_61 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/integrations/sdpa_attention.py:26 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_51: "bf16[1, 16, 1, 2048, 256][8388608, 256, 8388608, 4096, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_48, 2);  permute_48 = None
        expand_16: "bf16[1, 16, 2, 2048, 256][8388608, 256, 0, 4096, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_51, [1, 16, 2, 2048, 256]);  unsqueeze_51 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/integrations/sdpa_attention.py:27 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_12: "bf16[1, 16, 2, 2048, 256][16777216, 1048576, 524288, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_91: "bf16[1, 32, 2048, 256][16777216, 524288, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_12, [1, 32, 2048, 256]);  clone_12 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1232 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states
        view_88: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_152, [2048, 5376]);  convert_element_type_152 = None
        permute_47: "bf16[5376, 4096][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        mm_30: "bf16[2048, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_88, permute_47);  view_88 = permute_47 = None
        view_89: "bf16[1, 2048, 4096][8388608, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_30, [1, 2048, 4096]);  mm_30 = None
        view_90: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_89, [1, 2048, -1, 256]);  view_89 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:220 in forward, code: normed_output = self._norm(hidden_states.float())
        convert_element_type_165: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_90, torch.float32);  view_90 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:215 in _norm, code: mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        pow_63: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_165, 2)
        mean_31: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_63, [-1], True);  pow_63 = None
        add_62: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_31, 1e-06);  mean_31 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:217 in _norm, code: return hidden_states * torch.pow(mean_squared, -0.5)
        pow_64: "f32[1, 2048, 16, 1][32768, 16, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_62, -0.5);  add_62 = None
        mul_117: "f32[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_165, pow_64);  convert_element_type_165 = pow_64 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:223 in forward, code: return normed_output.type_as(hidden_states)
        convert_element_type_166: "bf16[1, 2048, 16, 256][8388608, 4096, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_117, torch.bfloat16);  mul_117 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1239 in forward, code: value_states = value_states.transpose(1, 2)
        permute_49: "bf16[1, 16, 2048, 256][8388608, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_166, [0, 2, 1, 3]);  convert_element_type_166 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/integrations/sdpa_attention.py:26 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_52: "bf16[1, 16, 1, 2048, 256][8388608, 256, 8388608, 4096, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_49, 2);  permute_49 = None
        expand_17: "bf16[1, 16, 2, 2048, 256][8388608, 256, 0, 4096, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_52, [1, 16, 2, 2048, 256]);  unsqueeze_52 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/integrations/sdpa_attention.py:27 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_13: "bf16[1, 16, 2, 2048, 256][16777216, 1048576, 524288, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_92: "bf16[1, 32, 2048, 256][16777216, 524288, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_13, [1, 32, 2048, 256]);  clone_13 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/integrations/sdpa_attention.py:174 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_12: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_11: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4: "bf16[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.where.self(expand_1, full_default_12, full_default_11);  expand_1 = full_default_12 = full_default_11 = None
        expand_18: "bf16[1, 32, 2048, 2048][4194304, 0, 2048, 1]cuda:0" = torch.ops.aten.expand.default(where_4, [1, 32, 2048, 2048]);  where_4 = None
        _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_45, view_91, view_92, expand_18, False, scale = 1.0);  permute_45 = view_91 = view_92 = expand_18 = None
        getitem_16: "bf16[1, 32, 2048, 256][16777216, 256, 8192, 1]cuda:0" = _scaled_dot_product_efficient_attention_4[0];  _scaled_dot_product_efficient_attention_4 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/integrations/sdpa_attention.py:184 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_50: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3]);  getitem_16 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1262 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_93: "bf16[1, 2048, 8192][16777216, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(permute_50, [1, 2048, -1]);  permute_50 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1263 in forward, code: attn_output = self.o_proj(attn_output)
        view_94: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(view_93, [2048, 8192]);  view_93 = None
        permute_51: "bf16[8192, 5376][1, 8192]cuda:0" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        mm_31: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.mm.default(view_94, permute_51);  view_94 = permute_51 = None
        view_95: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.reshape.default(mm_31, [1, 2048, 5376]);  mm_31 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:220 in forward, code: normed_output = self._norm(hidden_states.float())
        convert_element_type_169: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_95, torch.float32);  view_95 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:215 in _norm, code: mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        pow_65: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_169, 2)
        mean_32: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_65, [-1], True);  pow_65 = None
        add_63: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_32, 1e-06);  mean_32 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:217 in _norm, code: return hidden_states * torch.pow(mean_squared, -0.5)
        pow_66: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_63, -0.5);  add_63 = None
        mul_118: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_169, pow_66);  convert_element_type_169 = pow_66 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:222 in forward, code: normed_output = normed_output * self.weight.float()
        convert_element_type_170: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg68_1, torch.float32);  arg68_1 = None
        mul_119: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_118, convert_element_type_170);  mul_118 = convert_element_type_170 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:223 in forward, code: return normed_output.type_as(hidden_states)
        convert_element_type_171: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_119, torch.bfloat16);  mul_119 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1397 in forward, code: hidden_states = residual + hidden_states
        add_64: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_106, convert_element_type_171);  mul_106 = convert_element_type_171 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:220 in forward, code: normed_output = self._norm(hidden_states.float())
        convert_element_type_172: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_64, torch.float32)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:215 in _norm, code: mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        pow_67: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_172, 2)
        mean_33: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_67, [-1], True);  pow_67 = None
        add_65: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_33, 1e-06);  mean_33 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:217 in _norm, code: return hidden_states * torch.pow(mean_squared, -0.5)
        pow_68: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_65, -0.5);  add_65 = None
        mul_120: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_172, pow_68);  convert_element_type_172 = pow_68 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:222 in forward, code: normed_output = normed_output * self.weight.float()
        convert_element_type_173: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg69_1, torch.float32);  arg69_1 = None
        mul_121: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_120, convert_element_type_173);  mul_120 = convert_element_type_173 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:223 in forward, code: return normed_output.type_as(hidden_states)
        convert_element_type_174: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_121, torch.bfloat16);  mul_121 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1075 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_96: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_174, [2048, 5376])
        permute_52: "bf16[5376, 21504][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
        mm_32: "bf16[2048, 21504][21504, 1]cuda:0" = torch.ops.aten.mm.default(view_96, permute_52);  view_96 = permute_52 = None
        view_97: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.reshape.default(mm_32, [1, 2048, 21504]);  mm_32 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/activations.py:51 in forward, code: return self.act(input)
        convert_element_type_177: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_97, torch.float32);  view_97 = None
        mul_126: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_177, 0.5)
        mul_122: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_177, convert_element_type_177)
        mul_123: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_122, convert_element_type_177);  mul_122 = None
        mul_124: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_123, 0.044715);  mul_123 = None
        add_66: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_177, mul_124);  convert_element_type_177 = mul_124 = None
        mul_125: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_66, 0.7978845608028654);  add_66 = None
        tanh_4: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.tanh.default(mul_125);  mul_125 = None
        add_67: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_4, 1);  tanh_4 = None
        mul_127: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_126, add_67);  mul_126 = add_67 = None
        convert_element_type_178: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_127, torch.bfloat16);  mul_127 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1075 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_98: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_174, [2048, 5376]);  convert_element_type_174 = None
        permute_53: "bf16[5376, 21504][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        mm_33: "bf16[2048, 21504][21504, 1]cuda:0" = torch.ops.aten.mm.default(view_98, permute_53);  view_98 = permute_53 = None
        view_99: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.reshape.default(mm_33, [1, 2048, 21504]);  mm_33 = None
        mul_128: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_178, view_99);  convert_element_type_178 = view_99 = None
        view_100: "bf16[2048, 21504][21504, 1]cuda:0" = torch.ops.aten.reshape.default(mul_128, [2048, 21504]);  mul_128 = None
        permute_54: "bf16[21504, 5376][1, 21504]cuda:0" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        mm_34: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.mm.default(view_100, permute_54);  view_100 = permute_54 = None
        view_101: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.reshape.default(mm_34, [1, 2048, 5376]);  mm_34 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:220 in forward, code: normed_output = self._norm(hidden_states.float())
        convert_element_type_183: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_101, torch.float32);  view_101 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:215 in _norm, code: mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        pow_69: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_183, 2)
        mean_34: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_69, [-1], True);  pow_69 = None
        add_68: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_34, 1e-06);  mean_34 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:217 in _norm, code: return hidden_states * torch.pow(mean_squared, -0.5)
        pow_70: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_68, -0.5);  add_68 = None
        mul_129: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_183, pow_70);  convert_element_type_183 = pow_70 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:222 in forward, code: normed_output = normed_output * self.weight.float()
        convert_element_type_184: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg73_1, torch.float32);  arg73_1 = None
        mul_130: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_129, convert_element_type_184);  mul_129 = convert_element_type_184 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:223 in forward, code: return normed_output.type_as(hidden_states)
        convert_element_type_185: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_130, torch.bfloat16);  mul_130 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1418 in forward, code: hidden_states = residual + hidden_states
        add_69: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.add.Tensor(add_64, convert_element_type_185);  add_64 = convert_element_type_185 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1429 in forward, code: hidden_states *= self.layer_scalar
        mul_131: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_69, arg74_1);  add_69 = arg74_1 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:220 in forward, code: normed_output = self._norm(hidden_states.float())
        convert_element_type_186: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_131, torch.float32)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:215 in _norm, code: mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        pow_71: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_186, 2)
        mean_35: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_71, [-1], True);  pow_71 = None
        add_70: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_35, 1e-06);  mean_35 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:217 in _norm, code: return hidden_states * torch.pow(mean_squared, -0.5)
        pow_72: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_70, -0.5);  add_70 = None
        mul_132: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_186, pow_72);  convert_element_type_186 = pow_72 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:222 in forward, code: normed_output = normed_output * self.weight.float()
        convert_element_type_187: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg75_1, torch.float32);  arg75_1 = None
        mul_133: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_132, convert_element_type_187);  mul_132 = convert_element_type_187 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:223 in forward, code: return normed_output.type_as(hidden_states)
        convert_element_type_188: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_133, torch.bfloat16);  mul_133 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1217 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape)
        view_102: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_188, [2048, 5376])
        permute_55: "bf16[5376, 16384][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        mm_35: "bf16[2048, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_102, permute_55);  view_102 = permute_55 = None
        view_103: "bf16[1, 2048, 16384][33554432, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_35, [1, 2048, 16384]);  mm_35 = None
        view_104: "bf16[1, 2048, 32, 512][33554432, 16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_103, [1, 2048, -1, 512]);  view_103 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:220 in forward, code: normed_output = self._norm(hidden_states.float())
        convert_element_type_191: "f32[1, 2048, 32, 512][33554432, 16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_104, torch.float32);  view_104 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:215 in _norm, code: mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        pow_73: "f32[1, 2048, 32, 512][33554432, 16384, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_191, 2)
        mean_36: "f32[1, 2048, 32, 1][65536, 32, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_73, [-1], True);  pow_73 = None
        add_71: "f32[1, 2048, 32, 1][65536, 32, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_36, 1e-06);  mean_36 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:217 in _norm, code: return hidden_states * torch.pow(mean_squared, -0.5)
        pow_74: "f32[1, 2048, 32, 1][65536, 32, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_71, -0.5);  add_71 = None
        mul_134: "f32[1, 2048, 32, 512][33554432, 16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_191, pow_74);  convert_element_type_191 = pow_74 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:222 in forward, code: normed_output = normed_output * self.weight.float()
        convert_element_type_192: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg77_1, torch.float32);  arg77_1 = None
        mul_135: "f32[1, 2048, 32, 512][33554432, 16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_134, convert_element_type_192);  mul_134 = convert_element_type_192 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:223 in forward, code: return normed_output.type_as(hidden_states)
        convert_element_type_193: "bf16[1, 2048, 32, 512][33554432, 16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_135, torch.bfloat16);  mul_135 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1143 in forward, code: freqs = position_ids[..., None].float() * inv_freq.to(device=x.device, dtype=torch.float)
        unsqueeze_19: "i64[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, 2)
        convert_element_type: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(unsqueeze_19, torch.float32);  unsqueeze_19 = None
        mul_1: "f32[1, 2048, 256][524288, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type, arg3_1);  convert_element_type = arg3_1 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1144 in forward, code: emb = torch.cat((freqs, freqs), dim=-1)
        unsqueeze_20: "f32[1, 2048, 1, 256][524288, 256, 256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1, 2);  mul_1 = None
        expand_2: "f32[1, 2048, 2, 256][524288, 256, 0, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_20, [1, 2048, 2, 256]);  unsqueeze_20 = None
        clone: "f32[1, 2048, 2, 256][1048576, 512, 256, 1]cuda:0" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        view: "f32[1, 2048, 512][1048576, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone, [1, 2048, 512]);  clone = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1145 in forward, code: cos = emb.cos() * attention_scaling
        cos: "f32[1, 2048, 512][1048576, 512, 1]cuda:0" = torch.ops.aten.cos.default(view)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1148 in forward, code: return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        convert_element_type_1: "bf16[1, 2048, 512][1048576, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cos, torch.bfloat16);  cos = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:796 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_53: "bf16[1, 2048, 1, 512][1048576, 512, 512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 2)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:798 in apply_rotary_pos_emb, code: return (x * cos) + (rotate_half(x) * sin)
        mul_136: "bf16[1, 2048, 32, 512][33554432, 16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_193, unsqueeze_53);  unsqueeze_53 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:775 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_28: "bf16[1, 2048, 32, 256][33554432, 16384, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_193, 3, 256, 9223372036854775807)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:776 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_10: "bf16[1, 2048, 32, 256][16777216, 8192, 256, 1]cuda:0" = torch.ops.aten.neg.default(slice_28);  slice_28 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:774 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_27: "bf16[1, 2048, 32, 256][33554432, 16384, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_193, 3, 0, 256);  convert_element_type_193 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:776 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_12: "bf16[1, 2048, 32, 512][33554432, 16384, 512, 1]cuda:0" = torch.ops.aten.cat.default([neg_10, slice_27], -1);  neg_10 = slice_27 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1146 in forward, code: sin = emb.sin() * attention_scaling
        sin: "f32[1, 2048, 512][1048576, 512, 1]cuda:0" = torch.ops.aten.sin.default(view);  view = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1148 in forward, code: return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        convert_element_type_2: "bf16[1, 2048, 512][1048576, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sin, torch.bfloat16);  sin = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:797 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_54: "bf16[1, 2048, 1, 512][1048576, 512, 512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 2)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:798 in apply_rotary_pos_emb, code: return (x * cos) + (rotate_half(x) * sin)
        mul_137: "bf16[1, 2048, 32, 512][33554432, 16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_12, unsqueeze_54);  cat_12 = unsqueeze_54 = None
        add_72: "bf16[1, 2048, 32, 512][33554432, 16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_136, mul_137);  mul_136 = mul_137 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1220 in forward, code: query_states = query_states.transpose(1, 2)
        permute_56: "bf16[1, 32, 2048, 512][33554432, 512, 16384, 1]cuda:0" = torch.ops.aten.permute.default(add_72, [0, 2, 1, 3]);  add_72 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1231 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape)
        view_105: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_188, [2048, 5376]);  convert_element_type_188 = None
        permute_57: "bf16[5376, 2048][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
        mm_36: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_105, permute_57);  view_105 = permute_57 = None
        view_106: "bf16[1, 2048, 2048][4194304, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_36, [1, 2048, 2048]);  mm_36 = None
        view_107: "bf16[1, 2048, 4, 512][4194304, 2048, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_106, [1, 2048, -1, 512]);  view_106 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:220 in forward, code: normed_output = self._norm(hidden_states.float())
        convert_element_type_196: "f32[1, 2048, 4, 512][4194304, 2048, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_107, torch.float32)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:215 in _norm, code: mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        pow_75: "f32[1, 2048, 4, 512][4194304, 2048, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_196, 2)
        mean_37: "f32[1, 2048, 4, 1][8192, 4, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_75, [-1], True);  pow_75 = None
        add_73: "f32[1, 2048, 4, 1][8192, 4, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_37, 1e-06);  mean_37 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:217 in _norm, code: return hidden_states * torch.pow(mean_squared, -0.5)
        pow_76: "f32[1, 2048, 4, 1][8192, 4, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_73, -0.5);  add_73 = None
        mul_138: "f32[1, 2048, 4, 512][4194304, 2048, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_196, pow_76);  convert_element_type_196 = pow_76 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:222 in forward, code: normed_output = normed_output * self.weight.float()
        convert_element_type_197: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg79_1, torch.float32);  arg79_1 = None
        mul_139: "f32[1, 2048, 4, 512][4194304, 2048, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_138, convert_element_type_197);  mul_138 = convert_element_type_197 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:223 in forward, code: return normed_output.type_as(hidden_states)
        convert_element_type_198: "bf16[1, 2048, 4, 512][4194304, 2048, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_139, torch.bfloat16);  mul_139 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:796 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_55: "bf16[1, 2048, 1, 512][1048576, 512, 512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 2);  convert_element_type_1 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:798 in apply_rotary_pos_emb, code: return (x * cos) + (rotate_half(x) * sin)
        mul_140: "bf16[1, 2048, 4, 512][4194304, 2048, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_198, unsqueeze_55);  unsqueeze_55 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:775 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_30: "bf16[1, 2048, 4, 256][4194304, 2048, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_198, 3, 256, 9223372036854775807)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:776 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_11: "bf16[1, 2048, 4, 256][2097152, 1024, 256, 1]cuda:0" = torch.ops.aten.neg.default(slice_30);  slice_30 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:774 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_29: "bf16[1, 2048, 4, 256][4194304, 2048, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_198, 3, 0, 256);  convert_element_type_198 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:776 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_13: "bf16[1, 2048, 4, 512][4194304, 2048, 512, 1]cuda:0" = torch.ops.aten.cat.default([neg_11, slice_29], -1);  neg_11 = slice_29 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:797 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_56: "bf16[1, 2048, 1, 512][1048576, 512, 512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 2);  convert_element_type_2 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:798 in apply_rotary_pos_emb, code: return (x * cos) + (rotate_half(x) * sin)
        mul_141: "bf16[1, 2048, 4, 512][4194304, 2048, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(cat_13, unsqueeze_56);  cat_13 = unsqueeze_56 = None
        add_74: "bf16[1, 2048, 4, 512][4194304, 2048, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_140, mul_141);  mul_140 = mul_141 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1236 in forward, code: key_states = key_states.transpose(1, 2)
        permute_58: "bf16[1, 4, 2048, 512][4194304, 512, 2048, 1]cuda:0" = torch.ops.aten.permute.default(add_74, [0, 2, 1, 3]);  add_74 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/integrations/sdpa_attention.py:26 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_57: "bf16[1, 4, 1, 2048, 512][4194304, 512, 4194304, 2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_58, 2);  permute_58 = None
        expand_19: "bf16[1, 4, 8, 2048, 512][4194304, 512, 0, 2048, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_57, [1, 4, 8, 2048, 512]);  unsqueeze_57 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/integrations/sdpa_attention.py:27 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_14: "bf16[1, 4, 8, 2048, 512][33554432, 8388608, 1048576, 512, 1]cuda:0" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        view_108: "bf16[1, 32, 2048, 512][33554432, 1048576, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_14, [1, 32, 2048, 512]);  clone_14 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:220 in forward, code: normed_output = self._norm(hidden_states.float())
        convert_element_type_199: "f32[1, 2048, 4, 512][4194304, 2048, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_107, torch.float32);  view_107 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:215 in _norm, code: mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        pow_77: "f32[1, 2048, 4, 512][4194304, 2048, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_199, 2)
        mean_38: "f32[1, 2048, 4, 1][8192, 4, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_77, [-1], True);  pow_77 = None
        add_75: "f32[1, 2048, 4, 1][8192, 4, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_38, 1e-06);  mean_38 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:217 in _norm, code: return hidden_states * torch.pow(mean_squared, -0.5)
        pow_78: "f32[1, 2048, 4, 1][8192, 4, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_75, -0.5);  add_75 = None
        mul_142: "f32[1, 2048, 4, 512][4194304, 2048, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_199, pow_78);  convert_element_type_199 = pow_78 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:223 in forward, code: return normed_output.type_as(hidden_states)
        convert_element_type_200: "bf16[1, 2048, 4, 512][4194304, 2048, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_142, torch.bfloat16);  mul_142 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1239 in forward, code: value_states = value_states.transpose(1, 2)
        permute_59: "bf16[1, 4, 2048, 512][4194304, 512, 2048, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_200, [0, 2, 1, 3]);  convert_element_type_200 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/integrations/sdpa_attention.py:26 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_58: "bf16[1, 4, 1, 2048, 512][4194304, 512, 4194304, 2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(permute_59, 2);  permute_59 = None
        expand_20: "bf16[1, 4, 8, 2048, 512][4194304, 512, 0, 2048, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_58, [1, 4, 8, 2048, 512]);  unsqueeze_58 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/integrations/sdpa_attention.py:27 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_15: "bf16[1, 4, 8, 2048, 512][33554432, 8388608, 1048576, 512, 1]cuda:0" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_109: "bf16[1, 32, 2048, 512][33554432, 1048576, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_15, [1, 32, 2048, 512]);  clone_15 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/masking_utils.py:50 in and_mask, code: result = q_idx.new_ones((), dtype=torch.bool)
        full_default: "b8[][]cuda:0" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/masking_utils.py:510 in sdpa_mask, code: kv_arange = torch.arange(kv_length, device=device) + kv_offset
        iota_4: "i64[2048][1]cuda:0" = torch.ops.prims.iota.default(2048, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/masking_utils.py:370 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_7: "i64[1, 2048][2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_4, 0);  iota_4 = None
        unsqueeze_8: "i64[1, 1, 2048][2048, 2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 1);  unsqueeze_7 = None
        unsqueeze_9: "i64[1, 1, 1, 2048][2048, 2048, 2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_8, 2);  unsqueeze_8 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/masking_utils.py:509 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_3: "i64[2048][1]cuda:0" = torch.ops.prims.iota.default(2048, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/masking_utils.py:369 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_4: "i64[1, 2048][2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_3, 0);  iota_3 = None
        unsqueeze_5: "i64[1, 1, 2048][2048, 2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 1);  unsqueeze_4 = None
        unsqueeze_6: "i64[1, 1, 2048, 1][2048, 2048, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 3);  unsqueeze_5 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/masking_utils.py:76 in causal_mask_function, code: return kv_idx <= q_idx
        le: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.le.Tensor(unsqueeze_9, unsqueeze_6)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/masking_utils.py:52 in and_mask, code: result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        bitwise_and: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(full_default, le);  full_default = le = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/masking_utils.py:735 in find_packed_sequence_indices, code: first_dummy_value = position_ids[:, :1] - 1  # We just need the diff on this first value to be 1
        slice_1: "i64[1, 1][2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(unsqueeze, 1, 0, 1)
        sub: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.sub.Tensor(slice_1, 1);  slice_1 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/masking_utils.py:736 in find_packed_sequence_indices, code: position_diff = torch.diff(position_ids, prepend=first_dummy_value, dim=-1)
        cat: "i64[1, 2049][2049, 1]cuda:0" = torch.ops.aten.cat.default([sub, unsqueeze], -1);  sub = unsqueeze = None
        slice_3: "i64[1, 2048][2049, 1]cuda:0" = torch.ops.aten.slice.Tensor(cat, -1, 1, 2049)
        slice_2: "i64[1, 2048][2049, 1]cuda:0" = torch.ops.aten.slice.Tensor(cat, -1, 0, 2048);  cat = None
        sub_1: "i64[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(slice_3, slice_2);  slice_3 = slice_2 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/masking_utils.py:737 in find_packed_sequence_indices, code: packed_sequence_mask = (position_diff != 1).cumsum(-1)
        ne: "b8[1, 2048][2048, 1]cuda:0" = torch.ops.aten.ne.Scalar(sub_1, 1);  sub_1 = None
        cumsum: "i64[1, 2048][2048, 1]cuda:0" = torch.ops.aten.cumsum.default(ne, -1);  ne = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/masking_utils.py:507 in sdpa_mask, code: batch_arange = torch.arange(batch_size, device=device)
        iota_1: "i64[1][1]cuda:0" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/masking_utils.py:367 in _non_vmap_expansion_sdpa, code: batch_indices = batch_indices[:, None, None, None]
        unsqueeze_1: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_1, 1);  iota_1 = None
        unsqueeze_2: "i64[1, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 2);  unsqueeze_1 = None
        unsqueeze_3: "i64[1, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 3);  unsqueeze_2 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/masking_utils.py:184 in inner_mask, code: return packed_sequence_mask[batch_idx, q_idx] == packed_sequence_mask[batch_idx, kv_idx]
        index: "i64[1, 1, 2048, 1][2048, 2048, 1, 1]cuda:0" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_3, unsqueeze_6]);  unsqueeze_6 = None
        index_1: "i64[1, 1, 1, 2048][2048, 2048, 2048, 1]cuda:0" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_3, unsqueeze_9]);  cumsum = unsqueeze_3 = unsqueeze_9 = None
        eq: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/masking_utils.py:52 in and_mask, code: result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        bitwise_and_1: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, eq);  bitwise_and = eq = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/masking_utils.py:518 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.expand.default(bitwise_and_1, [1, -1, 2048, 2048]);  bitwise_and_1 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/integrations/sdpa_attention.py:174 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_14: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_13: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_5: "bf16[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.where.self(expand, full_default_14, full_default_13);  expand = full_default_14 = full_default_13 = None
        expand_21: "bf16[1, 32, 2048, 2048][4194304, 0, 2048, 1]cuda:0" = torch.ops.aten.expand.default(where_5, [1, 32, 2048, 2048]);  where_5 = None
        _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_56, view_108, view_109, expand_21, False, scale = 1.0);  permute_56 = view_108 = view_109 = expand_21 = None
        getitem_20: "bf16[1, 32, 2048, 512][33554432, 512, 16384, 1]cuda:0" = _scaled_dot_product_efficient_attention_5[0];  _scaled_dot_product_efficient_attention_5 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/integrations/sdpa_attention.py:184 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_60: "bf16[1, 2048, 32, 512][33554432, 16384, 512, 1]cuda:0" = torch.ops.aten.permute.default(getitem_20, [0, 2, 1, 3]);  getitem_20 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1262 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_110: "bf16[1, 2048, 16384][33554432, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_60, [1, 2048, -1]);  permute_60 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1263 in forward, code: attn_output = self.o_proj(attn_output)
        view_111: "bf16[2048, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(view_110, [2048, 16384]);  view_110 = None
        permute_61: "bf16[16384, 5376][1, 16384]cuda:0" = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
        mm_37: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.mm.default(view_111, permute_61);  view_111 = permute_61 = None
        view_112: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.reshape.default(mm_37, [1, 2048, 5376]);  mm_37 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:220 in forward, code: normed_output = self._norm(hidden_states.float())
        convert_element_type_203: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_112, torch.float32);  view_112 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:215 in _norm, code: mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        pow_79: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_203, 2)
        mean_39: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_79, [-1], True);  pow_79 = None
        add_76: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_39, 1e-06);  mean_39 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:217 in _norm, code: return hidden_states * torch.pow(mean_squared, -0.5)
        pow_80: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_76, -0.5);  add_76 = None
        mul_143: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_203, pow_80);  convert_element_type_203 = pow_80 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:222 in forward, code: normed_output = normed_output * self.weight.float()
        convert_element_type_204: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg81_1, torch.float32);  arg81_1 = None
        mul_144: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_143, convert_element_type_204);  mul_143 = convert_element_type_204 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:223 in forward, code: return normed_output.type_as(hidden_states)
        convert_element_type_205: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_144, torch.bfloat16);  mul_144 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1397 in forward, code: hidden_states = residual + hidden_states
        add_77: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_131, convert_element_type_205);  mul_131 = convert_element_type_205 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:220 in forward, code: normed_output = self._norm(hidden_states.float())
        convert_element_type_206: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_77, torch.float32)

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:215 in _norm, code: mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        pow_81: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_206, 2)
        mean_40: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_81, [-1], True);  pow_81 = None
        add_78: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_40, 1e-06);  mean_40 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:217 in _norm, code: return hidden_states * torch.pow(mean_squared, -0.5)
        pow_82: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_78, -0.5);  add_78 = None
        mul_145: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_206, pow_82);  convert_element_type_206 = pow_82 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:222 in forward, code: normed_output = normed_output * self.weight.float()
        convert_element_type_207: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg82_1, torch.float32);  arg82_1 = None
        mul_146: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_145, convert_element_type_207);  mul_145 = convert_element_type_207 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:223 in forward, code: return normed_output.type_as(hidden_states)
        convert_element_type_208: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_146, torch.bfloat16);  mul_146 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1075 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_113: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_208, [2048, 5376])
        permute_62: "bf16[5376, 21504][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        mm_38: "bf16[2048, 21504][21504, 1]cuda:0" = torch.ops.aten.mm.default(view_113, permute_62);  view_113 = permute_62 = None
        view_114: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.reshape.default(mm_38, [1, 2048, 21504]);  mm_38 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/activations.py:51 in forward, code: return self.act(input)
        convert_element_type_211: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_114, torch.float32);  view_114 = None
        mul_151: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_211, 0.5)
        mul_147: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_211, convert_element_type_211)
        mul_148: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_147, convert_element_type_211);  mul_147 = None
        mul_149: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_148, 0.044715);  mul_148 = None
        add_79: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_211, mul_149);  convert_element_type_211 = mul_149 = None
        mul_150: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_79, 0.7978845608028654);  add_79 = None
        tanh_5: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.tanh.default(mul_150);  mul_150 = None
        add_80: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_5, 1);  tanh_5 = None
        mul_152: "f32[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_151, add_80);  mul_151 = add_80 = None
        convert_element_type_212: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_152, torch.bfloat16);  mul_152 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1075 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        view_115: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_208, [2048, 5376]);  convert_element_type_208 = None
        permute_63: "bf16[5376, 21504][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
        mm_39: "bf16[2048, 21504][21504, 1]cuda:0" = torch.ops.aten.mm.default(view_115, permute_63);  view_115 = permute_63 = None
        view_116: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.reshape.default(mm_39, [1, 2048, 21504]);  mm_39 = None
        mul_153: "bf16[1, 2048, 21504][44040192, 21504, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_212, view_116);  convert_element_type_212 = view_116 = None
        view_117: "bf16[2048, 21504][21504, 1]cuda:0" = torch.ops.aten.reshape.default(mul_153, [2048, 21504]);  mul_153 = None
        permute_64: "bf16[21504, 5376][1, 21504]cuda:0" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        mm_40: "bf16[2048, 5376][5376, 1]cuda:0" = torch.ops.aten.mm.default(view_117, permute_64);  view_117 = permute_64 = None
        view_118: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.reshape.default(mm_40, [1, 2048, 5376]);  mm_40 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:220 in forward, code: normed_output = self._norm(hidden_states.float())
        convert_element_type_217: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_118, torch.float32);  view_118 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:215 in _norm, code: mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        pow_83: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_217, 2)
        mean_41: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_83, [-1], True);  pow_83 = None
        add_81: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_41, 1e-06);  mean_41 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:217 in _norm, code: return hidden_states * torch.pow(mean_squared, -0.5)
        pow_84: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_81, -0.5);  add_81 = None
        mul_154: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_217, pow_84);  convert_element_type_217 = pow_84 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:222 in forward, code: normed_output = normed_output * self.weight.float()
        convert_element_type_218: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg86_1, torch.float32);  arg86_1 = None
        mul_155: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_154, convert_element_type_218);  mul_154 = convert_element_type_218 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:223 in forward, code: return normed_output.type_as(hidden_states)
        convert_element_type_219: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_155, torch.bfloat16);  mul_155 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1418 in forward, code: hidden_states = residual + hidden_states
        add_82: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.add.Tensor(add_77, convert_element_type_219);  add_77 = convert_element_type_219 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1429 in forward, code: hidden_states *= self.layer_scalar
        mul_156: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_82, arg87_1);  add_82 = arg87_1 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:220 in forward, code: normed_output = self._norm(hidden_states.float())
        convert_element_type_220: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_156, torch.float32);  mul_156 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:215 in _norm, code: mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        pow_85: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_220, 2)
        mean_42: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.mean.dim(pow_85, [-1], True);  pow_85 = None
        add_83: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(mean_42, 1e-06);  mean_42 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:217 in _norm, code: return hidden_states * torch.pow(mean_squared, -0.5)
        pow_86: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_83, -0.5);  add_83 = None
        mul_157: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_220, pow_86);  convert_element_type_220 = pow_86 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:222 in forward, code: normed_output = normed_output * self.weight.float()
        convert_element_type_221: "f32[5376][1]cuda:0" = torch.ops.prims.convert_element_type.default(arg88_1, torch.float32);  arg88_1 = None
        mul_158: "f32[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_157, convert_element_type_221);  mul_157 = convert_element_type_221 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:223 in forward, code: return normed_output.type_as(hidden_states)
        convert_element_type_222: "bf16[1, 2048, 5376][11010048, 5376, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_158, torch.bfloat16);  mul_158 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/modeling_utils.py:5035 in slice_logits_to_keep, code: return hidden_states[:, logits_to_keep, :]
        slice_31: "bf16[1, 1, 5376][11010048, 5376, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_222, 1, -1, 9223372036854775807);  convert_element_type_222 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1853 in forward, code: logits = self.lm_head(hidden_states)
        view_119: "bf16[1, 5376][11010048, 1]cuda:0" = torch.ops.aten.reshape.default(slice_31, [1, 5376]);  slice_31 = None
        permute_65: "bf16[5376, 262144][1, 5376]cuda:0" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        mm_41: "bf16[1, 262144][262144, 1]cuda:0" = torch.ops.aten.mm.default(view_119, permute_65);  view_119 = permute_65 = None
        view_120: "bf16[1, 1, 262144][262144, 262144, 1]cuda:0" = torch.ops.aten.reshape.default(mm_41, [1, 1, 262144]);  mm_41 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1855 in forward, code: logits = logits / self.config.final_logit_softcapping
        div: "bf16[1, 1, 262144][262144, 262144, 1]cuda:0" = torch.ops.aten.div.Tensor(view_120, 30.0);  view_120 = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1856 in forward, code: logits = torch.tanh(logits)
        tanh_6: "bf16[1, 1, 262144][262144, 262144, 1]cuda:0" = torch.ops.aten.tanh.default(div);  div = None

        # File: /workspaces/pytorch-devcontainers/transformers/src/transformers/models/gemma4/modeling_gemma4.py:1857 in forward, code: logits = logits * self.config.final_logit_softcapping
        mul_159: "bf16[1, 1, 262144][262144, 262144, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_6, 30.0);  tanh_6 = None
        return (mul_159,)

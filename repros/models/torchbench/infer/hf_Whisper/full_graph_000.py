import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f16[8, 80, 3000]", arg1_1: "f16[384, 80, 3]", arg2_1: "f16[384]", arg3_1: "f16[384, 384, 3]", arg4_1: "f16[384]", arg5_1: "f16[1500, 384]", arg6_1: "f16[384]", arg7_1: "f16[384]", arg8_1: "f16[384, 384]", arg9_1: "f16[384]", arg10_1: "f16[384, 384]", arg11_1: "f16[384, 384]", arg12_1: "f16[384]", arg13_1: "f16[384, 384]", arg14_1: "f16[384]", arg15_1: "f16[384]", arg16_1: "f16[384]", arg17_1: "f16[1536, 384]", arg18_1: "f16[1536]", arg19_1: "f16[384, 1536]", arg20_1: "f16[384]", arg21_1: "f16[384]", arg22_1: "f16[384]", arg23_1: "f16[384, 384]", arg24_1: "f16[384]", arg25_1: "f16[384, 384]", arg26_1: "f16[384, 384]", arg27_1: "f16[384]", arg28_1: "f16[384, 384]", arg29_1: "f16[384]", arg30_1: "f16[384]", arg31_1: "f16[384]", arg32_1: "f16[1536, 384]", arg33_1: "f16[1536]", arg34_1: "f16[384, 1536]", arg35_1: "f16[384]", arg36_1: "f16[384]", arg37_1: "f16[384]", arg38_1: "f16[384, 384]", arg39_1: "f16[384]", arg40_1: "f16[384, 384]", arg41_1: "f16[384, 384]", arg42_1: "f16[384]", arg43_1: "f16[384, 384]", arg44_1: "f16[384]", arg45_1: "f16[384]", arg46_1: "f16[384]", arg47_1: "f16[1536, 384]", arg48_1: "f16[1536]", arg49_1: "f16[384, 1536]", arg50_1: "f16[384]", arg51_1: "f16[384]", arg52_1: "f16[384]", arg53_1: "f16[384, 384]", arg54_1: "f16[384]", arg55_1: "f16[384, 384]", arg56_1: "f16[384, 384]", arg57_1: "f16[384]", arg58_1: "f16[384, 384]", arg59_1: "f16[384]", arg60_1: "f16[384]", arg61_1: "f16[384]", arg62_1: "f16[1536, 384]", arg63_1: "f16[1536]", arg64_1: "f16[384, 1536]", arg65_1: "f16[384]", arg66_1: "f16[384]", arg67_1: "f16[384]", arg68_1: "f16[256, 384]", arg69_1: "f16[256]", arg70_1: "f16[2, 256]", arg71_1: "f16[2]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:618 in forward, code: inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        convolution: "f16[8, 384, 3000]" = torch.ops.aten.convolution.default(arg0_1, arg1_1, arg2_1, [1], [1], [1], False, [0], 1);  arg0_1 = arg1_1 = arg2_1 = None
        convert_element_type: "f32[8, 384, 3000]" = torch.ops.prims.convert_element_type.default(convolution, torch.float32);  convolution = None
        mul: "f32[8, 384, 3000]" = torch.ops.aten.mul.Tensor(convert_element_type, 0.5)
        mul_1: "f32[8, 384, 3000]" = torch.ops.aten.mul.Tensor(convert_element_type, 0.7071067811865476);  convert_element_type = None
        erf: "f32[8, 384, 3000]" = torch.ops.aten.erf.default(mul_1);  mul_1 = None
        add: "f32[8, 384, 3000]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_2: "f32[8, 384, 3000]" = torch.ops.aten.mul.Tensor(mul, add);  mul = add = None
        convert_element_type_1: "f16[8, 384, 3000]" = torch.ops.prims.convert_element_type.default(mul_2, torch.float16);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:619 in forward, code: inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))
        convolution_1: "f16[8, 384, 1500]" = torch.ops.aten.convolution.default(convert_element_type_1, arg3_1, arg4_1, [2], [1], [1], False, [0], 1);  convert_element_type_1 = arg3_1 = arg4_1 = None
        convert_element_type_2: "f32[8, 384, 1500]" = torch.ops.prims.convert_element_type.default(convolution_1, torch.float32);  convolution_1 = None
        mul_3: "f32[8, 384, 1500]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 0.5)
        mul_4: "f32[8, 384, 1500]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 0.7071067811865476);  convert_element_type_2 = None
        erf_1: "f32[8, 384, 1500]" = torch.ops.aten.erf.default(mul_4);  mul_4 = None
        add_1: "f32[8, 384, 1500]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_5: "f32[8, 384, 1500]" = torch.ops.aten.mul.Tensor(mul_3, add_1);  mul_3 = add_1 = None
        convert_element_type_3: "f16[8, 384, 1500]" = torch.ops.prims.convert_element_type.default(mul_5, torch.float16);  mul_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:621 in forward, code: inputs_embeds = inputs_embeds.permute(0, 2, 1)
        permute: "f16[8, 1500, 384]" = torch.ops.aten.permute.default(convert_element_type_3, [0, 2, 1]);  convert_element_type_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:622 in forward, code: all_positions = torch.arange(self.embed_positions.num_embeddings, device=inputs_embeds.device)
        iota: "i64[1500]" = torch.ops.prims.iota.default(1500, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:624 in forward, code: hidden_states = inputs_embeds + self.embed_positions(all_positions)
        embedding: "f16[1500, 384]" = torch.ops.aten.embedding.default(arg5_1, iota);  arg5_1 = iota = None
        add_2: "f16[8, 1500, 384]" = torch.ops.aten.add.Tensor(permute, embedding);  permute = embedding = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:392 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        clone_1: "f16[8, 1500, 384]" = torch.ops.aten.clone.default(add_2, memory_format = torch.contiguous_format)
        convert_element_type_4: "f32[8, 1500, 384]" = torch.ops.prims.convert_element_type.default(clone_1, torch.float32);  clone_1 = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_4, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 1500, 1]" = var_mean[0]
        getitem_1: "f32[8, 1500, 1]" = var_mean[1];  var_mean = None
        sub: "f32[8, 1500, 384]" = torch.ops.aten.sub.Tensor(convert_element_type_4, getitem_1);  convert_element_type_4 = getitem_1 = None
        add_3: "f32[8, 1500, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[8, 1500, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        mul_6: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_7: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(mul_6, arg6_1);  mul_6 = arg6_1 = None
        add_4: "f32[8, 1500, 384]" = torch.ops.aten.add.Tensor(mul_7, arg7_1);  mul_7 = arg7_1 = None
        convert_element_type_5: "f16[8, 1500, 384]" = torch.ops.prims.convert_element_type.default(add_4, torch.float16);  add_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        view: "f16[12000, 384]" = torch.ops.aten.reshape.default(convert_element_type_5, [12000, 384])
        permute_1: "f16[384, 384]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm: "f16[12000, 384]" = torch.ops.aten.addmm.default(arg9_1, view, permute_1);  arg9_1 = view = permute_1 = None
        view_1: "f16[8, 1500, 384]" = torch.ops.aten.reshape.default(addmm, [8, 1500, 384]);  addmm = None
        mul_8: "f16[8, 1500, 384]" = torch.ops.aten.mul.Tensor(view_1, 0.125);  view_1 = None
        view_2: "f16[8, 1500, 6, 64]" = torch.ops.aten.reshape.default(mul_8, [8, 1500, -1, 64]);  mul_8 = None
        permute_2: "f16[8, 6, 1500, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        clone_2: "f16[8, 6, 1500, 64]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_3: "f16[12000, 384]" = torch.ops.aten.reshape.default(convert_element_type_5, [12000, 384])
        permute_3: "f16[384, 384]" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        mm: "f16[12000, 384]" = torch.ops.aten.mm.default(view_3, permute_3);  view_3 = permute_3 = None
        view_4: "f16[8, 1500, 384]" = torch.ops.aten.reshape.default(mm, [8, 1500, 384]);  mm = None
        view_5: "f16[8, 1500, 6, 64]" = torch.ops.aten.reshape.default(view_4, [8, -1, 6, 64]);  view_4 = None
        permute_4: "f16[8, 6, 1500, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        clone_3: "f16[8, 6, 1500, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_6: "f16[12000, 384]" = torch.ops.aten.reshape.default(convert_element_type_5, [12000, 384]);  convert_element_type_5 = None
        permute_5: "f16[384, 384]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_1: "f16[12000, 384]" = torch.ops.aten.addmm.default(arg12_1, view_6, permute_5);  arg12_1 = view_6 = permute_5 = None
        view_7: "f16[8, 1500, 384]" = torch.ops.aten.reshape.default(addmm_1, [8, 1500, 384]);  addmm_1 = None
        view_8: "f16[8, 1500, 6, 64]" = torch.ops.aten.reshape.default(view_7, [8, -1, 6, 64]);  view_7 = None
        permute_6: "f16[8, 6, 1500, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        clone_4: "f16[8, 6, 1500, 64]" = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format);  permute_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention = torch.ops.aten._scaled_dot_product_cudnn_attention.default(clone_2, clone_3, clone_4, None, False, scale = 1.0);  clone_2 = clone_3 = clone_4 = None
        getitem_2: "f16[8, 6, 1500, 64]" = _scaled_dot_product_cudnn_attention[0];  _scaled_dot_product_cudnn_attention = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "f16[8, 1500, 6, 64]" = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3]);  getitem_2 = None
        clone_5: "f16[8, 1500, 6, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:353 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_9: "f16[8, 1500, 384]" = torch.ops.aten.reshape.default(clone_5, [8, 1500, -1]);  clone_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        view_10: "f16[12000, 384]" = torch.ops.aten.reshape.default(view_9, [12000, 384]);  view_9 = None
        permute_8: "f16[384, 384]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_2: "f16[12000, 384]" = torch.ops.aten.addmm.default(arg14_1, view_10, permute_8);  arg14_1 = view_10 = permute_8 = None
        view_11: "f16[8, 1500, 384]" = torch.ops.aten.reshape.default(addmm_2, [8, 1500, 384]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:399 in forward, code: hidden_states = residual + hidden_states
        add_5: "f16[8, 1500, 384]" = torch.ops.aten.add.Tensor(add_2, view_11);  add_2 = view_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:402 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        clone_7: "f16[8, 1500, 384]" = torch.ops.aten.clone.default(add_5, memory_format = torch.contiguous_format)
        convert_element_type_17: "f32[8, 1500, 384]" = torch.ops.prims.convert_element_type.default(clone_7, torch.float32);  clone_7 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_17, [2], correction = 0, keepdim = True)
        getitem_11: "f32[8, 1500, 1]" = var_mean_1[0]
        getitem_12: "f32[8, 1500, 1]" = var_mean_1[1];  var_mean_1 = None
        sub_1: "f32[8, 1500, 384]" = torch.ops.aten.sub.Tensor(convert_element_type_17, getitem_12);  convert_element_type_17 = getitem_12 = None
        add_6: "f32[8, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_11, 1e-05);  getitem_11 = None
        rsqrt_1: "f32[8, 1500, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        mul_9: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        mul_10: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(mul_9, arg15_1);  mul_9 = arg15_1 = None
        add_7: "f32[8, 1500, 384]" = torch.ops.aten.add.Tensor(mul_10, arg16_1);  mul_10 = arg16_1 = None
        convert_element_type_18: "f16[8, 1500, 384]" = torch.ops.prims.convert_element_type.default(add_7, torch.float16);  add_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:403 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_12: "f16[12000, 384]" = torch.ops.aten.reshape.default(convert_element_type_18, [12000, 384]);  convert_element_type_18 = None
        permute_9: "f16[384, 1536]" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        addmm_3: "f16[12000, 1536]" = torch.ops.aten.addmm.default(arg18_1, view_12, permute_9);  arg18_1 = view_12 = permute_9 = None
        view_13: "f16[8, 1500, 1536]" = torch.ops.aten.reshape.default(addmm_3, [8, 1500, 1536]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_22: "f32[8, 1500, 1536]" = torch.ops.prims.convert_element_type.default(view_13, torch.float32);  view_13 = None
        mul_11: "f32[8, 1500, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 0.5)
        mul_12: "f32[8, 1500, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 0.7071067811865476);  convert_element_type_22 = None
        erf_2: "f32[8, 1500, 1536]" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_8: "f32[8, 1500, 1536]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_13: "f32[8, 1500, 1536]" = torch.ops.aten.mul.Tensor(mul_11, add_8);  mul_11 = add_8 = None
        convert_element_type_23: "f16[8, 1500, 1536]" = torch.ops.prims.convert_element_type.default(mul_13, torch.float16);  mul_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:405 in forward, code: hidden_states = self.fc2(hidden_states)
        view_14: "f16[12000, 1536]" = torch.ops.aten.reshape.default(convert_element_type_23, [12000, 1536]);  convert_element_type_23 = None
        permute_10: "f16[1536, 384]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        addmm_4: "f16[12000, 384]" = torch.ops.aten.addmm.default(arg20_1, view_14, permute_10);  arg20_1 = view_14 = permute_10 = None
        view_15: "f16[8, 1500, 384]" = torch.ops.aten.reshape.default(addmm_4, [8, 1500, 384]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:407 in forward, code: hidden_states = residual + hidden_states
        add_9: "f16[8, 1500, 384]" = torch.ops.aten.add.Tensor(add_5, view_15);  add_5 = view_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:411 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_27: "f32[8, 1500, 384]" = torch.ops.prims.convert_element_type.default(add_9, torch.float32);  add_9 = None
        clamp_min: "f32[8, 1500, 384]" = torch.ops.aten.clamp_min.default(convert_element_type_27, -64504.0);  convert_element_type_27 = None
        clamp_max: "f32[8, 1500, 384]" = torch.ops.aten.clamp_max.default(clamp_min, 64504.0);  clamp_min = None
        convert_element_type_28: "f16[8, 1500, 384]" = torch.ops.prims.convert_element_type.default(clamp_max, torch.float16);  clamp_max = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:392 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        clone_10: "f16[8, 1500, 384]" = torch.ops.aten.clone.default(convert_element_type_28, memory_format = torch.contiguous_format)
        convert_element_type_29: "f32[8, 1500, 384]" = torch.ops.prims.convert_element_type.default(clone_10, torch.float32);  clone_10 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_29, [2], correction = 0, keepdim = True)
        getitem_13: "f32[8, 1500, 1]" = var_mean_2[0]
        getitem_14: "f32[8, 1500, 1]" = var_mean_2[1];  var_mean_2 = None
        sub_2: "f32[8, 1500, 384]" = torch.ops.aten.sub.Tensor(convert_element_type_29, getitem_14);  convert_element_type_29 = getitem_14 = None
        add_10: "f32[8, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_13, 1e-05);  getitem_13 = None
        rsqrt_2: "f32[8, 1500, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        mul_14: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
        mul_15: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(mul_14, arg21_1);  mul_14 = arg21_1 = None
        add_11: "f32[8, 1500, 384]" = torch.ops.aten.add.Tensor(mul_15, arg22_1);  mul_15 = arg22_1 = None
        convert_element_type_30: "f16[8, 1500, 384]" = torch.ops.prims.convert_element_type.default(add_11, torch.float16);  add_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        view_16: "f16[12000, 384]" = torch.ops.aten.reshape.default(convert_element_type_30, [12000, 384])
        permute_11: "f16[384, 384]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        addmm_5: "f16[12000, 384]" = torch.ops.aten.addmm.default(arg24_1, view_16, permute_11);  arg24_1 = view_16 = permute_11 = None
        view_17: "f16[8, 1500, 384]" = torch.ops.aten.reshape.default(addmm_5, [8, 1500, 384]);  addmm_5 = None
        mul_16: "f16[8, 1500, 384]" = torch.ops.aten.mul.Tensor(view_17, 0.125);  view_17 = None
        view_18: "f16[8, 1500, 6, 64]" = torch.ops.aten.reshape.default(mul_16, [8, 1500, -1, 64]);  mul_16 = None
        permute_12: "f16[8, 6, 1500, 64]" = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None
        clone_11: "f16[8, 6, 1500, 64]" = torch.ops.aten.clone.default(permute_12, memory_format = torch.contiguous_format);  permute_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_19: "f16[12000, 384]" = torch.ops.aten.reshape.default(convert_element_type_30, [12000, 384])
        permute_13: "f16[384, 384]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        mm_1: "f16[12000, 384]" = torch.ops.aten.mm.default(view_19, permute_13);  view_19 = permute_13 = None
        view_20: "f16[8, 1500, 384]" = torch.ops.aten.reshape.default(mm_1, [8, 1500, 384]);  mm_1 = None
        view_21: "f16[8, 1500, 6, 64]" = torch.ops.aten.reshape.default(view_20, [8, -1, 6, 64]);  view_20 = None
        permute_14: "f16[8, 6, 1500, 64]" = torch.ops.aten.permute.default(view_21, [0, 2, 1, 3]);  view_21 = None
        clone_12: "f16[8, 6, 1500, 64]" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_22: "f16[12000, 384]" = torch.ops.aten.reshape.default(convert_element_type_30, [12000, 384]);  convert_element_type_30 = None
        permute_15: "f16[384, 384]" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        addmm_6: "f16[12000, 384]" = torch.ops.aten.addmm.default(arg27_1, view_22, permute_15);  arg27_1 = view_22 = permute_15 = None
        view_23: "f16[8, 1500, 384]" = torch.ops.aten.reshape.default(addmm_6, [8, 1500, 384]);  addmm_6 = None
        view_24: "f16[8, 1500, 6, 64]" = torch.ops.aten.reshape.default(view_23, [8, -1, 6, 64]);  view_23 = None
        permute_16: "f16[8, 6, 1500, 64]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
        clone_13: "f16[8, 6, 1500, 64]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_1 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(clone_11, clone_12, clone_13, None, False, scale = 1.0);  clone_11 = clone_12 = clone_13 = None
        getitem_15: "f16[8, 6, 1500, 64]" = _scaled_dot_product_cudnn_attention_1[0];  _scaled_dot_product_cudnn_attention_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_17: "f16[8, 1500, 6, 64]" = torch.ops.aten.permute.default(getitem_15, [0, 2, 1, 3]);  getitem_15 = None
        clone_14: "f16[8, 1500, 6, 64]" = torch.ops.aten.clone.default(permute_17, memory_format = torch.contiguous_format);  permute_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:353 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_25: "f16[8, 1500, 384]" = torch.ops.aten.reshape.default(clone_14, [8, 1500, -1]);  clone_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        view_26: "f16[12000, 384]" = torch.ops.aten.reshape.default(view_25, [12000, 384]);  view_25 = None
        permute_18: "f16[384, 384]" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        addmm_7: "f16[12000, 384]" = torch.ops.aten.addmm.default(arg29_1, view_26, permute_18);  arg29_1 = view_26 = permute_18 = None
        view_27: "f16[8, 1500, 384]" = torch.ops.aten.reshape.default(addmm_7, [8, 1500, 384]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:399 in forward, code: hidden_states = residual + hidden_states
        add_12: "f16[8, 1500, 384]" = torch.ops.aten.add.Tensor(convert_element_type_28, view_27);  convert_element_type_28 = view_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:402 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        clone_16: "f16[8, 1500, 384]" = torch.ops.aten.clone.default(add_12, memory_format = torch.contiguous_format)
        convert_element_type_42: "f32[8, 1500, 384]" = torch.ops.prims.convert_element_type.default(clone_16, torch.float32);  clone_16 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_42, [2], correction = 0, keepdim = True)
        getitem_24: "f32[8, 1500, 1]" = var_mean_3[0]
        getitem_25: "f32[8, 1500, 1]" = var_mean_3[1];  var_mean_3 = None
        sub_3: "f32[8, 1500, 384]" = torch.ops.aten.sub.Tensor(convert_element_type_42, getitem_25);  convert_element_type_42 = getitem_25 = None
        add_13: "f32[8, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_3: "f32[8, 1500, 1]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        mul_17: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = rsqrt_3 = None
        mul_18: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(mul_17, arg30_1);  mul_17 = arg30_1 = None
        add_14: "f32[8, 1500, 384]" = torch.ops.aten.add.Tensor(mul_18, arg31_1);  mul_18 = arg31_1 = None
        convert_element_type_43: "f16[8, 1500, 384]" = torch.ops.prims.convert_element_type.default(add_14, torch.float16);  add_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:403 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_28: "f16[12000, 384]" = torch.ops.aten.reshape.default(convert_element_type_43, [12000, 384]);  convert_element_type_43 = None
        permute_19: "f16[384, 1536]" = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
        addmm_8: "f16[12000, 1536]" = torch.ops.aten.addmm.default(arg33_1, view_28, permute_19);  arg33_1 = view_28 = permute_19 = None
        view_29: "f16[8, 1500, 1536]" = torch.ops.aten.reshape.default(addmm_8, [8, 1500, 1536]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_47: "f32[8, 1500, 1536]" = torch.ops.prims.convert_element_type.default(view_29, torch.float32);  view_29 = None
        mul_19: "f32[8, 1500, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_47, 0.5)
        mul_20: "f32[8, 1500, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_47, 0.7071067811865476);  convert_element_type_47 = None
        erf_3: "f32[8, 1500, 1536]" = torch.ops.aten.erf.default(mul_20);  mul_20 = None
        add_15: "f32[8, 1500, 1536]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_21: "f32[8, 1500, 1536]" = torch.ops.aten.mul.Tensor(mul_19, add_15);  mul_19 = add_15 = None
        convert_element_type_48: "f16[8, 1500, 1536]" = torch.ops.prims.convert_element_type.default(mul_21, torch.float16);  mul_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:405 in forward, code: hidden_states = self.fc2(hidden_states)
        view_30: "f16[12000, 1536]" = torch.ops.aten.reshape.default(convert_element_type_48, [12000, 1536]);  convert_element_type_48 = None
        permute_20: "f16[1536, 384]" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        addmm_9: "f16[12000, 384]" = torch.ops.aten.addmm.default(arg35_1, view_30, permute_20);  arg35_1 = view_30 = permute_20 = None
        view_31: "f16[8, 1500, 384]" = torch.ops.aten.reshape.default(addmm_9, [8, 1500, 384]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:407 in forward, code: hidden_states = residual + hidden_states
        add_16: "f16[8, 1500, 384]" = torch.ops.aten.add.Tensor(add_12, view_31);  add_12 = view_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:411 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_52: "f32[8, 1500, 384]" = torch.ops.prims.convert_element_type.default(add_16, torch.float32);  add_16 = None
        clamp_min_1: "f32[8, 1500, 384]" = torch.ops.aten.clamp_min.default(convert_element_type_52, -64504.0);  convert_element_type_52 = None
        clamp_max_1: "f32[8, 1500, 384]" = torch.ops.aten.clamp_max.default(clamp_min_1, 64504.0);  clamp_min_1 = None
        convert_element_type_53: "f16[8, 1500, 384]" = torch.ops.prims.convert_element_type.default(clamp_max_1, torch.float16);  clamp_max_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:392 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        clone_19: "f16[8, 1500, 384]" = torch.ops.aten.clone.default(convert_element_type_53, memory_format = torch.contiguous_format)
        convert_element_type_54: "f32[8, 1500, 384]" = torch.ops.prims.convert_element_type.default(clone_19, torch.float32);  clone_19 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_54, [2], correction = 0, keepdim = True)
        getitem_26: "f32[8, 1500, 1]" = var_mean_4[0]
        getitem_27: "f32[8, 1500, 1]" = var_mean_4[1];  var_mean_4 = None
        sub_4: "f32[8, 1500, 384]" = torch.ops.aten.sub.Tensor(convert_element_type_54, getitem_27);  convert_element_type_54 = getitem_27 = None
        add_17: "f32[8, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_4: "f32[8, 1500, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        mul_22: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = rsqrt_4 = None
        mul_23: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(mul_22, arg36_1);  mul_22 = arg36_1 = None
        add_18: "f32[8, 1500, 384]" = torch.ops.aten.add.Tensor(mul_23, arg37_1);  mul_23 = arg37_1 = None
        convert_element_type_55: "f16[8, 1500, 384]" = torch.ops.prims.convert_element_type.default(add_18, torch.float16);  add_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        view_32: "f16[12000, 384]" = torch.ops.aten.reshape.default(convert_element_type_55, [12000, 384])
        permute_21: "f16[384, 384]" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        addmm_10: "f16[12000, 384]" = torch.ops.aten.addmm.default(arg39_1, view_32, permute_21);  arg39_1 = view_32 = permute_21 = None
        view_33: "f16[8, 1500, 384]" = torch.ops.aten.reshape.default(addmm_10, [8, 1500, 384]);  addmm_10 = None
        mul_24: "f16[8, 1500, 384]" = torch.ops.aten.mul.Tensor(view_33, 0.125);  view_33 = None
        view_34: "f16[8, 1500, 6, 64]" = torch.ops.aten.reshape.default(mul_24, [8, 1500, -1, 64]);  mul_24 = None
        permute_22: "f16[8, 6, 1500, 64]" = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None
        clone_20: "f16[8, 6, 1500, 64]" = torch.ops.aten.clone.default(permute_22, memory_format = torch.contiguous_format);  permute_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_35: "f16[12000, 384]" = torch.ops.aten.reshape.default(convert_element_type_55, [12000, 384])
        permute_23: "f16[384, 384]" = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
        mm_2: "f16[12000, 384]" = torch.ops.aten.mm.default(view_35, permute_23);  view_35 = permute_23 = None
        view_36: "f16[8, 1500, 384]" = torch.ops.aten.reshape.default(mm_2, [8, 1500, 384]);  mm_2 = None
        view_37: "f16[8, 1500, 6, 64]" = torch.ops.aten.reshape.default(view_36, [8, -1, 6, 64]);  view_36 = None
        permute_24: "f16[8, 6, 1500, 64]" = torch.ops.aten.permute.default(view_37, [0, 2, 1, 3]);  view_37 = None
        clone_21: "f16[8, 6, 1500, 64]" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_38: "f16[12000, 384]" = torch.ops.aten.reshape.default(convert_element_type_55, [12000, 384]);  convert_element_type_55 = None
        permute_25: "f16[384, 384]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        addmm_11: "f16[12000, 384]" = torch.ops.aten.addmm.default(arg42_1, view_38, permute_25);  arg42_1 = view_38 = permute_25 = None
        view_39: "f16[8, 1500, 384]" = torch.ops.aten.reshape.default(addmm_11, [8, 1500, 384]);  addmm_11 = None
        view_40: "f16[8, 1500, 6, 64]" = torch.ops.aten.reshape.default(view_39, [8, -1, 6, 64]);  view_39 = None
        permute_26: "f16[8, 6, 1500, 64]" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
        clone_22: "f16[8, 6, 1500, 64]" = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_2 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(clone_20, clone_21, clone_22, None, False, scale = 1.0);  clone_20 = clone_21 = clone_22 = None
        getitem_28: "f16[8, 6, 1500, 64]" = _scaled_dot_product_cudnn_attention_2[0];  _scaled_dot_product_cudnn_attention_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_27: "f16[8, 1500, 6, 64]" = torch.ops.aten.permute.default(getitem_28, [0, 2, 1, 3]);  getitem_28 = None
        clone_23: "f16[8, 1500, 6, 64]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:353 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_41: "f16[8, 1500, 384]" = torch.ops.aten.reshape.default(clone_23, [8, 1500, -1]);  clone_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        view_42: "f16[12000, 384]" = torch.ops.aten.reshape.default(view_41, [12000, 384]);  view_41 = None
        permute_28: "f16[384, 384]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_12: "f16[12000, 384]" = torch.ops.aten.addmm.default(arg44_1, view_42, permute_28);  arg44_1 = view_42 = permute_28 = None
        view_43: "f16[8, 1500, 384]" = torch.ops.aten.reshape.default(addmm_12, [8, 1500, 384]);  addmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:399 in forward, code: hidden_states = residual + hidden_states
        add_19: "f16[8, 1500, 384]" = torch.ops.aten.add.Tensor(convert_element_type_53, view_43);  convert_element_type_53 = view_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:402 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        clone_25: "f16[8, 1500, 384]" = torch.ops.aten.clone.default(add_19, memory_format = torch.contiguous_format)
        convert_element_type_67: "f32[8, 1500, 384]" = torch.ops.prims.convert_element_type.default(clone_25, torch.float32);  clone_25 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_67, [2], correction = 0, keepdim = True)
        getitem_37: "f32[8, 1500, 1]" = var_mean_5[0]
        getitem_38: "f32[8, 1500, 1]" = var_mean_5[1];  var_mean_5 = None
        sub_5: "f32[8, 1500, 384]" = torch.ops.aten.sub.Tensor(convert_element_type_67, getitem_38);  convert_element_type_67 = getitem_38 = None
        add_20: "f32[8, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_37, 1e-05);  getitem_37 = None
        rsqrt_5: "f32[8, 1500, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        mul_25: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = rsqrt_5 = None
        mul_26: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(mul_25, arg45_1);  mul_25 = arg45_1 = None
        add_21: "f32[8, 1500, 384]" = torch.ops.aten.add.Tensor(mul_26, arg46_1);  mul_26 = arg46_1 = None
        convert_element_type_68: "f16[8, 1500, 384]" = torch.ops.prims.convert_element_type.default(add_21, torch.float16);  add_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:403 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_44: "f16[12000, 384]" = torch.ops.aten.reshape.default(convert_element_type_68, [12000, 384]);  convert_element_type_68 = None
        permute_29: "f16[384, 1536]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_13: "f16[12000, 1536]" = torch.ops.aten.addmm.default(arg48_1, view_44, permute_29);  arg48_1 = view_44 = permute_29 = None
        view_45: "f16[8, 1500, 1536]" = torch.ops.aten.reshape.default(addmm_13, [8, 1500, 1536]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_72: "f32[8, 1500, 1536]" = torch.ops.prims.convert_element_type.default(view_45, torch.float32);  view_45 = None
        mul_27: "f32[8, 1500, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_72, 0.5)
        mul_28: "f32[8, 1500, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_72, 0.7071067811865476);  convert_element_type_72 = None
        erf_4: "f32[8, 1500, 1536]" = torch.ops.aten.erf.default(mul_28);  mul_28 = None
        add_22: "f32[8, 1500, 1536]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_29: "f32[8, 1500, 1536]" = torch.ops.aten.mul.Tensor(mul_27, add_22);  mul_27 = add_22 = None
        convert_element_type_73: "f16[8, 1500, 1536]" = torch.ops.prims.convert_element_type.default(mul_29, torch.float16);  mul_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:405 in forward, code: hidden_states = self.fc2(hidden_states)
        view_46: "f16[12000, 1536]" = torch.ops.aten.reshape.default(convert_element_type_73, [12000, 1536]);  convert_element_type_73 = None
        permute_30: "f16[1536, 384]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        addmm_14: "f16[12000, 384]" = torch.ops.aten.addmm.default(arg50_1, view_46, permute_30);  arg50_1 = view_46 = permute_30 = None
        view_47: "f16[8, 1500, 384]" = torch.ops.aten.reshape.default(addmm_14, [8, 1500, 384]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:407 in forward, code: hidden_states = residual + hidden_states
        add_23: "f16[8, 1500, 384]" = torch.ops.aten.add.Tensor(add_19, view_47);  add_19 = view_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:411 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_77: "f32[8, 1500, 384]" = torch.ops.prims.convert_element_type.default(add_23, torch.float32);  add_23 = None
        clamp_min_2: "f32[8, 1500, 384]" = torch.ops.aten.clamp_min.default(convert_element_type_77, -64504.0);  convert_element_type_77 = None
        clamp_max_2: "f32[8, 1500, 384]" = torch.ops.aten.clamp_max.default(clamp_min_2, 64504.0);  clamp_min_2 = None
        convert_element_type_78: "f16[8, 1500, 384]" = torch.ops.prims.convert_element_type.default(clamp_max_2, torch.float16);  clamp_max_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:392 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        clone_28: "f16[8, 1500, 384]" = torch.ops.aten.clone.default(convert_element_type_78, memory_format = torch.contiguous_format)
        convert_element_type_79: "f32[8, 1500, 384]" = torch.ops.prims.convert_element_type.default(clone_28, torch.float32);  clone_28 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_79, [2], correction = 0, keepdim = True)
        getitem_39: "f32[8, 1500, 1]" = var_mean_6[0]
        getitem_40: "f32[8, 1500, 1]" = var_mean_6[1];  var_mean_6 = None
        sub_6: "f32[8, 1500, 384]" = torch.ops.aten.sub.Tensor(convert_element_type_79, getitem_40);  convert_element_type_79 = getitem_40 = None
        add_24: "f32[8, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_39, 1e-05);  getitem_39 = None
        rsqrt_6: "f32[8, 1500, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        mul_30: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = rsqrt_6 = None
        mul_31: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(mul_30, arg51_1);  mul_30 = arg51_1 = None
        add_25: "f32[8, 1500, 384]" = torch.ops.aten.add.Tensor(mul_31, arg52_1);  mul_31 = arg52_1 = None
        convert_element_type_80: "f16[8, 1500, 384]" = torch.ops.prims.convert_element_type.default(add_25, torch.float16);  add_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        view_48: "f16[12000, 384]" = torch.ops.aten.reshape.default(convert_element_type_80, [12000, 384])
        permute_31: "f16[384, 384]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        addmm_15: "f16[12000, 384]" = torch.ops.aten.addmm.default(arg54_1, view_48, permute_31);  arg54_1 = view_48 = permute_31 = None
        view_49: "f16[8, 1500, 384]" = torch.ops.aten.reshape.default(addmm_15, [8, 1500, 384]);  addmm_15 = None
        mul_32: "f16[8, 1500, 384]" = torch.ops.aten.mul.Tensor(view_49, 0.125);  view_49 = None
        view_50: "f16[8, 1500, 6, 64]" = torch.ops.aten.reshape.default(mul_32, [8, 1500, -1, 64]);  mul_32 = None
        permute_32: "f16[8, 6, 1500, 64]" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
        clone_29: "f16[8, 6, 1500, 64]" = torch.ops.aten.clone.default(permute_32, memory_format = torch.contiguous_format);  permute_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_51: "f16[12000, 384]" = torch.ops.aten.reshape.default(convert_element_type_80, [12000, 384])
        permute_33: "f16[384, 384]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        mm_3: "f16[12000, 384]" = torch.ops.aten.mm.default(view_51, permute_33);  view_51 = permute_33 = None
        view_52: "f16[8, 1500, 384]" = torch.ops.aten.reshape.default(mm_3, [8, 1500, 384]);  mm_3 = None
        view_53: "f16[8, 1500, 6, 64]" = torch.ops.aten.reshape.default(view_52, [8, -1, 6, 64]);  view_52 = None
        permute_34: "f16[8, 6, 1500, 64]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
        clone_30: "f16[8, 6, 1500, 64]" = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format);  permute_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        view_54: "f16[12000, 384]" = torch.ops.aten.reshape.default(convert_element_type_80, [12000, 384]);  convert_element_type_80 = None
        permute_35: "f16[384, 384]" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        addmm_16: "f16[12000, 384]" = torch.ops.aten.addmm.default(arg57_1, view_54, permute_35);  arg57_1 = view_54 = permute_35 = None
        view_55: "f16[8, 1500, 384]" = torch.ops.aten.reshape.default(addmm_16, [8, 1500, 384]);  addmm_16 = None
        view_56: "f16[8, 1500, 6, 64]" = torch.ops.aten.reshape.default(view_55, [8, -1, 6, 64]);  view_55 = None
        permute_36: "f16[8, 6, 1500, 64]" = torch.ops.aten.permute.default(view_56, [0, 2, 1, 3]);  view_56 = None
        clone_31: "f16[8, 6, 1500, 64]" = torch.ops.aten.clone.default(permute_36, memory_format = torch.contiguous_format);  permute_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_3 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(clone_29, clone_30, clone_31, None, False, scale = 1.0);  clone_29 = clone_30 = clone_31 = None
        getitem_41: "f16[8, 6, 1500, 64]" = _scaled_dot_product_cudnn_attention_3[0];  _scaled_dot_product_cudnn_attention_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_37: "f16[8, 1500, 6, 64]" = torch.ops.aten.permute.default(getitem_41, [0, 2, 1, 3]);  getitem_41 = None
        clone_32: "f16[8, 1500, 6, 64]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:353 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_57: "f16[8, 1500, 384]" = torch.ops.aten.reshape.default(clone_32, [8, 1500, -1]);  clone_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        view_58: "f16[12000, 384]" = torch.ops.aten.reshape.default(view_57, [12000, 384]);  view_57 = None
        permute_38: "f16[384, 384]" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        addmm_17: "f16[12000, 384]" = torch.ops.aten.addmm.default(arg59_1, view_58, permute_38);  arg59_1 = view_58 = permute_38 = None
        view_59: "f16[8, 1500, 384]" = torch.ops.aten.reshape.default(addmm_17, [8, 1500, 384]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:399 in forward, code: hidden_states = residual + hidden_states
        add_26: "f16[8, 1500, 384]" = torch.ops.aten.add.Tensor(convert_element_type_78, view_59);  convert_element_type_78 = view_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:402 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        clone_34: "f16[8, 1500, 384]" = torch.ops.aten.clone.default(add_26, memory_format = torch.contiguous_format)
        convert_element_type_92: "f32[8, 1500, 384]" = torch.ops.prims.convert_element_type.default(clone_34, torch.float32);  clone_34 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_92, [2], correction = 0, keepdim = True)
        getitem_50: "f32[8, 1500, 1]" = var_mean_7[0]
        getitem_51: "f32[8, 1500, 1]" = var_mean_7[1];  var_mean_7 = None
        sub_7: "f32[8, 1500, 384]" = torch.ops.aten.sub.Tensor(convert_element_type_92, getitem_51);  convert_element_type_92 = getitem_51 = None
        add_27: "f32[8, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
        rsqrt_7: "f32[8, 1500, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        mul_33: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = rsqrt_7 = None
        mul_34: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(mul_33, arg60_1);  mul_33 = arg60_1 = None
        add_28: "f32[8, 1500, 384]" = torch.ops.aten.add.Tensor(mul_34, arg61_1);  mul_34 = arg61_1 = None
        convert_element_type_93: "f16[8, 1500, 384]" = torch.ops.prims.convert_element_type.default(add_28, torch.float16);  add_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:403 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_60: "f16[12000, 384]" = torch.ops.aten.reshape.default(convert_element_type_93, [12000, 384]);  convert_element_type_93 = None
        permute_39: "f16[384, 1536]" = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
        addmm_18: "f16[12000, 1536]" = torch.ops.aten.addmm.default(arg63_1, view_60, permute_39);  arg63_1 = view_60 = permute_39 = None
        view_61: "f16[8, 1500, 1536]" = torch.ops.aten.reshape.default(addmm_18, [8, 1500, 1536]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_97: "f32[8, 1500, 1536]" = torch.ops.prims.convert_element_type.default(view_61, torch.float32);  view_61 = None
        mul_35: "f32[8, 1500, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_97, 0.5)
        mul_36: "f32[8, 1500, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_97, 0.7071067811865476);  convert_element_type_97 = None
        erf_5: "f32[8, 1500, 1536]" = torch.ops.aten.erf.default(mul_36);  mul_36 = None
        add_29: "f32[8, 1500, 1536]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_37: "f32[8, 1500, 1536]" = torch.ops.aten.mul.Tensor(mul_35, add_29);  mul_35 = add_29 = None
        convert_element_type_98: "f16[8, 1500, 1536]" = torch.ops.prims.convert_element_type.default(mul_37, torch.float16);  mul_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:405 in forward, code: hidden_states = self.fc2(hidden_states)
        view_62: "f16[12000, 1536]" = torch.ops.aten.reshape.default(convert_element_type_98, [12000, 1536]);  convert_element_type_98 = None
        permute_40: "f16[1536, 384]" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        addmm_19: "f16[12000, 384]" = torch.ops.aten.addmm.default(arg65_1, view_62, permute_40);  arg65_1 = view_62 = permute_40 = None
        view_63: "f16[8, 1500, 384]" = torch.ops.aten.reshape.default(addmm_19, [8, 1500, 384]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:407 in forward, code: hidden_states = residual + hidden_states
        add_30: "f16[8, 1500, 384]" = torch.ops.aten.add.Tensor(add_26, view_63);  add_26 = view_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:411 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_102: "f32[8, 1500, 384]" = torch.ops.prims.convert_element_type.default(add_30, torch.float32);  add_30 = None
        clamp_min_3: "f32[8, 1500, 384]" = torch.ops.aten.clamp_min.default(convert_element_type_102, -64504.0);  convert_element_type_102 = None
        clamp_max_3: "f32[8, 1500, 384]" = torch.ops.aten.clamp_max.default(clamp_min_3, 64504.0);  clamp_min_3 = None
        convert_element_type_103: "f16[8, 1500, 384]" = torch.ops.prims.convert_element_type.default(clamp_max_3, torch.float16);  clamp_max_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:642 in forward, code: hidden_states = self.layer_norm(hidden_states)
        clone_37: "f16[8, 1500, 384]" = torch.ops.aten.clone.default(convert_element_type_103, memory_format = torch.contiguous_format);  convert_element_type_103 = None
        convert_element_type_104: "f32[8, 1500, 384]" = torch.ops.prims.convert_element_type.default(clone_37, torch.float32);  clone_37 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_104, [2], correction = 0, keepdim = True)
        getitem_52: "f32[8, 1500, 1]" = var_mean_8[0]
        getitem_53: "f32[8, 1500, 1]" = var_mean_8[1];  var_mean_8 = None
        sub_8: "f32[8, 1500, 384]" = torch.ops.aten.sub.Tensor(convert_element_type_104, getitem_53);  convert_element_type_104 = getitem_53 = None
        add_31: "f32[8, 1500, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
        rsqrt_8: "f32[8, 1500, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        mul_38: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = rsqrt_8 = None
        mul_39: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(mul_38, arg66_1);  mul_38 = arg66_1 = None
        add_32: "f32[8, 1500, 384]" = torch.ops.aten.add.Tensor(mul_39, arg67_1);  mul_39 = arg67_1 = None
        convert_element_type_105: "f16[8, 1500, 384]" = torch.ops.prims.convert_element_type.default(add_32, torch.float16);  add_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:1333 in forward, code: hidden_states = self.projector(hidden_states)
        view_64: "f16[12000, 384]" = torch.ops.aten.reshape.default(convert_element_type_105, [12000, 384]);  convert_element_type_105 = None
        permute_41: "f16[384, 256]" = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
        addmm_20: "f16[12000, 256]" = torch.ops.aten.addmm.default(arg69_1, view_64, permute_41);  arg69_1 = view_64 = permute_41 = None
        view_65: "f16[8, 1500, 256]" = torch.ops.aten.reshape.default(addmm_20, [8, 1500, 256]);  addmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:1334 in forward, code: pooled_output = hidden_states.mean(dim=1)
        mean: "f16[8, 256]" = torch.ops.aten.mean.dim(view_65, [1]);  view_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:1336 in forward, code: logits = self.classifier(pooled_output)
        permute_42: "f16[256, 2]" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
        addmm_21: "f16[8, 2]" = torch.ops.aten.addmm.default(arg71_1, mean, permute_42);  arg71_1 = mean = permute_42 = None
        return (addmm_21,)

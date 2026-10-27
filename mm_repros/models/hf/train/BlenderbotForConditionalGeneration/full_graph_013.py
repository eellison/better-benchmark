class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[2560][1]cuda:0", primals_3: "f32[16, 128, 2560][327680, 2560, 1]cuda:0", primals_13: "f32[2560][1]cuda:0", getitem_1: "f32[16, 128, 1][128, 1, 1]cuda:0", rsqrt: "f32[16, 128, 1][128, 1, 1]cuda:0", view: "bf16[2048, 2560][2560, 1]cuda:0", where_1: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0", view_16: "bf16[2048, 2560][2560, 1]cuda:0", addmm_3: "bf16[2048, 2560][2560, 1]cuda:0", gt: "b8[16, 128, 2560][327680, 2560, 1]cuda:0", getitem_3: "f32[16, 128, 1][128, 1, 1]cuda:0", rsqrt_1: "f32[16, 128, 1][128, 1, 1]cuda:0", view_18: "bf16[2048, 2560][2560, 1]cuda:0", addmm_4: "bf16[2048, 10240][10240, 1]cuda:0", view_20: "bf16[2048, 10240][10240, 1]cuda:0", gt_1: "b8[16, 128, 2560][327680, 2560, 1]cuda:0", permute_11: "bf16[2560, 10240][10240, 1]cuda:0", permute_15: "bf16[10240, 2560][2560, 1]cuda:0", permute_19: "bf16[2560, 2560][2560, 1]cuda:0", permute_25: "bf16[512, 80, 128][10240, 1, 80]cuda:0", permute_26: "bf16[512, 80, 128][10240, 1, 80]cuda:0", permute_27: "bf16[512, 128, 80][10240, 1, 128]cuda:0", permute_31: "bf16[2560, 2560][2560, 1]cuda:0", permute_35: "bf16[2560, 2560][2560, 1]cuda:0", permute_40: "bf16[2560, 2560][2560, 1]cuda:0", tangents_1: "f32[16, 128, 2560][327680, 2560, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:294 in forward, code: hidden_states = residual + hidden_states
        convert_element_type_42: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(tangents_1, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:293 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        convert_element_type_43: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_1, torch.bfloat16);  gt_1 = None
        mul_13: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_43, 1.1111111111111112);  convert_element_type_43 = None
        mul_14: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_42, mul_13);  convert_element_type_42 = mul_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:292 in forward, code: hidden_states = self.fc2(hidden_states)
        view_22: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(mul_14, [2048, 2560]);  mul_14 = None
        mm: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.mm.default(view_22, permute_11);  permute_11 = None
        permute_12: "bf16[2560, 2048][1, 2560]cuda:0" = torch.ops.aten.permute.default(view_22, [1, 0])
        mm_1: "bf16[2560, 10240][10240, 1]cuda:0" = torch.ops.aten.mm.default(permute_12, view_20);  permute_12 = view_20 = None
        sum_2: "f32[1, 2560][2560, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_22, [0], True, dtype = torch.float32);  view_22 = None
        view_23: "f32[2560][1]cuda:0" = torch.ops.aten.reshape.default(sum_2, [2560]);  sum_2 = None
        convert_element_type_48: "bf16[2560][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_23, torch.bfloat16);  view_23 = None
        view_24: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [16, 128, 10240]);  mm = None
        convert_element_type_49: "f32[2560, 10240][10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_50: "f32[2560][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_48, torch.float32);  convert_element_type_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_51: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_24, torch.float32);  view_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:290 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_19: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [16, 128, 10240]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_35: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_19, torch.float32);  view_19 = None
        mul_9: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_35, 0.7071067811865476)
        erf: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_9);  mul_9 = None
        add_6: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_16: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_6, 0.5);  add_6 = None
        mul_17: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_35, convert_element_type_35)
        mul_18: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_17, -0.5);  mul_17 = None
        exp_1: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.exp.default(mul_18);  mul_18 = None
        mul_19: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
        mul_20: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_35, mul_19);  convert_element_type_35 = mul_19 = None
        add_9: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_16, mul_20);  mul_16 = mul_20 = None
        mul_21: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_51, add_9);  convert_element_type_51 = add_9 = None
        convert_element_type_53: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_21, torch.bfloat16);  mul_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:290 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_25: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_53, [2048, 10240]);  convert_element_type_53 = None
        mm_2: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.mm.default(view_25, permute_15);  permute_15 = None
        permute_16: "bf16[10240, 2048][1, 10240]cuda:0" = torch.ops.aten.permute.default(view_25, [1, 0])
        mm_3: "bf16[10240, 2560][2560, 1]cuda:0" = torch.ops.aten.mm.default(permute_16, view_18);  permute_16 = view_18 = None
        sum_3: "f32[1, 10240][10240, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_25, [0], True, dtype = torch.float32);  view_25 = None
        view_26: "f32[10240][1]cuda:0" = torch.ops.aten.reshape.default(sum_3, [10240]);  sum_3 = None
        convert_element_type_58: "bf16[10240][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_26, torch.bfloat16);  view_26 = None
        view_27: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(mm_2, [16, 128, 2560]);  mm_2 = None
        convert_element_type_59: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_27, torch.float32);  view_27 = None
        convert_element_type_60: "f32[10240, 2560][2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_61: "f32[10240][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_58, torch.float32);  convert_element_type_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:289 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        mul_23: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_59, primals_13);  primals_13 = None
        mul_24: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_23, 2560)
        sum_4: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_23, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_17: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [16, 128, 2560]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:285 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        mul_4: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt, view_17);  view_17 = None
        mul_5: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4, 1.1111111111111112);  mul_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:286 in forward, code: hidden_states = residual + hidden_states
        add_3: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(primals_3, mul_5);  mul_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:289 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        sub_2: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_3, getitem_3);  add_3 = getitem_3 = None
        mul_6: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
        mul_25: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_23, mul_6);  mul_23 = None
        sum_5: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_25, [2], True);  mul_25 = None
        mul_26: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_6, sum_5);  sum_5 = None
        sub_4: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_24, sum_4);  mul_24 = sum_4 = None
        sub_5: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_4, mul_26);  sub_4 = mul_26 = None
        div_1: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_1, 2560);  rsqrt_1 = None
        mul_27: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_1, sub_5);  div_1 = sub_5 = None
        mul_28: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_59, mul_6);  mul_6 = None
        sum_6: "f32[2560][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_28, [0, 1]);  mul_28 = None
        sum_7: "f32[2560][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_59, [0, 1]);  convert_element_type_59 = None
        add_10: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(tangents_1, mul_27);  tangents_1 = mul_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:286 in forward, code: hidden_states = residual + hidden_states
        convert_element_type_62: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_10, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:285 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        convert_element_type_63: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt, torch.bfloat16);  gt = None
        mul_29: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_63, 1.1111111111111112);  convert_element_type_63 = None
        mul_30: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_62, mul_29);  convert_element_type_62 = mul_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_28: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(mul_30, [2048, 2560]);  mul_30 = None
        mm_4: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.mm.default(view_28, permute_19);  permute_19 = None
        permute_20: "bf16[2560, 2048][1, 2560]cuda:0" = torch.ops.aten.permute.default(view_28, [1, 0])
        mm_5: "bf16[2560, 2560][2560, 1]cuda:0" = torch.ops.aten.mm.default(permute_20, view_16);  permute_20 = view_16 = None
        sum_8: "f32[1, 2560][2560, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_28, [0], True, dtype = torch.float32);  view_28 = None
        view_29: "f32[2560][1]cuda:0" = torch.ops.aten.reshape.default(sum_8, [2560]);  sum_8 = None
        convert_element_type_68: "bf16[2560][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_29, torch.bfloat16);  view_29 = None
        view_30: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(mm_4, [16, 128, 2560]);  mm_4 = None
        convert_element_type_69: "f32[2560, 2560][2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_70: "f32[2560][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_68, torch.float32);  convert_element_type_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_31: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_30, [16, 128, 32, 80]);  view_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_23: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_7: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
        view_32: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_7, [512, 128, 80]);  clone_7 = None
        expand_2: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_1, [16, 32, 128, 128])
        view_12: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_2, [512, 128, 128]);  expand_2 = None
        permute_24: "bf16[512, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
        bmm_2: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.bmm.default(permute_24, view_32);  permute_24 = None
        bmm_3: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_32, permute_25);  view_32 = permute_25 = None
        view_33: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_2, [16, 32, 128, 80]);  bmm_2 = None
        view_34: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_3, [16, 32, 128, 128]);  bmm_3 = None
        convert_element_type_75: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_34, torch.float32);  view_34 = None
        convert_element_type_76: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_1, torch.float32);  where_1 = None
        mul_31: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_75, convert_element_type_76);  convert_element_type_75 = None
        sum_9: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_31, [-1], True)
        neg: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_76);  convert_element_type_76 = None
        fma: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg, sum_9, mul_31);  neg = sum_9 = mul_31 = None
        convert_element_type_77: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        view_35: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_77, [512, 128, 128]);  convert_element_type_77 = None
        bmm_4: "bf16[512, 80, 128][10240, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_26, view_35);  permute_26 = None
        bmm_5: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.bmm.default(view_35, permute_27);  view_35 = permute_27 = None
        view_36: "bf16[16, 32, 80, 128][327680, 10240, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_4, [16, 32, 80, 128]);  bmm_4 = None
        view_37: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_5, [16, 32, 128, 80]);  bmm_5 = None
        mul_32: "bf16[16, 32, 80, 128][327680, 10240, 128, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_36, 0.334370152488211);  view_36 = None
        permute_28: "bf16[16, 32, 128, 80][327680, 10240, 1, 128]cuda:0" = torch.ops.aten.permute.default(mul_32, [0, 1, 3, 2]);  mul_32 = None
        mul_33: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_37, 0.334370152488211);  view_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        permute_29: "bf16[16, 128, 32, 80][327680, 80, 10240, 1]cuda:0" = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3]);  view_33 = None
        clone_8: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_38: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(clone_8, [16, 128, 2560]);  clone_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        permute_30: "bf16[16, 128, 32, 80][327680, 1, 10240, 128]cuda:0" = torch.ops.aten.permute.default(permute_28, [0, 2, 1, 3]);  permute_28 = None
        view_39: "bf16[16, 128, 2560][327680, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_30, [16, 128, 2560]);  permute_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_40: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_38, [2048, 2560]);  view_38 = None
        mm_6: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.mm.default(view_40, permute_31);  permute_31 = None
        permute_32: "bf16[2560, 2048][1, 2560]cuda:0" = torch.ops.aten.permute.default(view_40, [1, 0])
        mm_7: "bf16[2560, 2560][2560, 1]cuda:0" = torch.ops.aten.mm.default(permute_32, view);  permute_32 = None
        sum_10: "f32[1, 2560][2560, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_40, [0], True, dtype = torch.float32);  view_40 = None
        view_41: "f32[2560][1]cuda:0" = torch.ops.aten.reshape.default(sum_10, [2560]);  sum_10 = None
        convert_element_type_86: "bf16[2560][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_41, torch.bfloat16);  view_41 = None
        view_42: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(mm_6, [16, 128, 2560]);  mm_6 = None
        convert_element_type_87: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_42, torch.float32);  view_42 = None
        convert_element_type_88: "f32[2560, 2560][2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_89: "f32[2560][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_86, torch.float32);  convert_element_type_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        clone_9: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.clone.default(view_39, memory_format = torch.contiguous_format);  view_39 = None
        view_43: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(clone_9, [2048, 2560]);  clone_9 = None
        mm_8: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.mm.default(view_43, permute_35);  permute_35 = None
        permute_36: "bf16[2560, 2048][1, 2560]cuda:0" = torch.ops.aten.permute.default(view_43, [1, 0])
        mm_9: "bf16[2560, 2560][2560, 1]cuda:0" = torch.ops.aten.mm.default(permute_36, view);  permute_36 = None
        sum_11: "f32[1, 2560][2560, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_43, [0], True, dtype = torch.float32);  view_43 = None
        view_44: "f32[2560][1]cuda:0" = torch.ops.aten.reshape.default(sum_11, [2560]);  sum_11 = None
        convert_element_type_94: "bf16[2560][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_44, torch.bfloat16);  view_44 = None
        view_45: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(mm_8, [16, 128, 2560]);  mm_8 = None
        convert_element_type_95: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_45, torch.float32);  view_45 = None
        add_11: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_87, convert_element_type_95);  convert_element_type_87 = convert_element_type_95 = None
        convert_element_type_96: "f32[2560, 2560][2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        convert_element_type_97: "f32[2560][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_94, torch.float32);  convert_element_type_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_39: "bf16[16, 128, 32, 80][327680, 80, 10240, 1]cuda:0" = torch.ops.aten.permute.default(mul_33, [0, 2, 1, 3]);  mul_33 = None
        clone_10: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.clone.default(permute_39, memory_format = torch.contiguous_format);  permute_39 = None
        view_46: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(clone_10, [16, 128, 2560]);  clone_10 = None
        view_47: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_46, [2048, 2560]);  view_46 = None
        mm_10: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.mm.default(view_47, permute_40);  permute_40 = None
        permute_41: "bf16[2560, 2048][1, 2560]cuda:0" = torch.ops.aten.permute.default(view_47, [1, 0])
        mm_11: "bf16[2560, 2560][2560, 1]cuda:0" = torch.ops.aten.mm.default(permute_41, view);  permute_41 = view = None
        sum_12: "f32[1, 2560][2560, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_47, [0], True, dtype = torch.float32);  view_47 = None
        view_48: "f32[2560][1]cuda:0" = torch.ops.aten.reshape.default(sum_12, [2560]);  sum_12 = None
        convert_element_type_102: "bf16[2560][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_48, torch.bfloat16);  view_48 = None
        view_49: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(mm_10, [16, 128, 2560]);  mm_10 = None
        convert_element_type_103: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_49, torch.float32);  view_49 = None
        add_12: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_11, convert_element_type_103);  add_11 = convert_element_type_103 = None
        convert_element_type_104: "f32[2560, 2560][2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_11, torch.float32);  mm_11 = None
        convert_element_type_105: "f32[2560][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_102, torch.float32);  convert_element_type_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:279 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        mul_35: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_12, primals_1);  primals_1 = None
        mul_36: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_35, 2560)
        sum_13: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_35, [2], True)
        sub: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(primals_3, getitem_1);  primals_3 = getitem_1 = None
        mul: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_37: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_35, mul);  mul_35 = None
        sum_14: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_37, [2], True);  mul_37 = None
        mul_38: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, sum_14);  sum_14 = None
        sub_7: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_36, sum_13);  mul_36 = sum_13 = None
        sub_8: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_7, mul_38);  sub_7 = mul_38 = None
        div_2: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt, 2560);  rsqrt = None
        mul_39: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_2, sub_8);  div_2 = sub_8 = None
        mul_40: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_12, mul);  mul = None
        sum_15: "f32[2560][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_40, [0, 1]);  mul_40 = None
        sum_16: "f32[2560][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_12, [0, 1]);  add_12 = None
        add_13: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_10, mul_39);  add_10 = mul_39 = None
        return (sum_15, sum_16, add_13, convert_element_type_104, convert_element_type_105, convert_element_type_96, convert_element_type_97, convert_element_type_88, convert_element_type_89, None, convert_element_type_69, convert_element_type_70, sum_6, sum_7, convert_element_type_60, convert_element_type_61, convert_element_type_49, convert_element_type_50)

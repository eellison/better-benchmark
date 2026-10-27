class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[2560][1]cuda:0", primals_3: "f32[16, 128, 2560][327680, 2560, 1]cuda:0", primals_14: "f32[2560][1]cuda:0", primals_25: "f32[2560][1]cuda:0", getitem_1: "f32[16, 128, 1][128, 1, 1]cuda:0", rsqrt: "f32[16, 128, 1][128, 1, 1]cuda:0", view: "bf16[2048, 2560][2560, 1]cuda:0", permute_1: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0", permute_4: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0", permute_5: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0", where: "bf16[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0", getitem_2: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0", getitem_3: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0", getitem_8: "i64[][]cuda:0", getitem_9: "i64[][]cuda:0", addmm_3: "bf16[2048, 2560][2560, 1]cuda:0", gt: "b8[16, 128, 2560][327680, 2560, 1]cuda:0", getitem_12: "f32[16, 128, 1][128, 1, 1]cuda:0", rsqrt_1: "f32[16, 128, 1][128, 1, 1]cuda:0", view_12: "bf16[2048, 2560][2560, 1]cuda:0", view_15: "bf16[2048, 2560][2560, 1]cuda:0", where_2: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0", view_28: "bf16[2048, 2560][2560, 1]cuda:0", gt_1: "b8[16, 128, 2560][327680, 2560, 1]cuda:0", mul_10: "f32[16, 128, 2560][327680, 2560, 1]cuda:0", view_30: "bf16[2048, 2560][2560, 1]cuda:0", addmm_8: "bf16[2048, 10240][10240, 1]cuda:0", view_32: "bf16[2048, 10240][10240, 1]cuda:0", gt_2: "b8[16, 128, 2560][327680, 2560, 1]cuda:0", permute_19: "bf16[2560, 10240][10240, 1]cuda:0", permute_23: "bf16[10240, 2560][2560, 1]cuda:0", div_1: "f32[16, 128, 1][128, 1, 1]cuda:0", permute_27: "bf16[2560, 2560][2560, 1]cuda:0", permute_33: "bf16[512, 80, 128][10240, 1, 80]cuda:0", permute_34: "bf16[512, 80, 128][10240, 1, 80]cuda:0", permute_35: "bf16[512, 128, 80][10240, 1, 128]cuda:0", permute_39: "bf16[2560, 2560][2560, 1]cuda:0", permute_43: "bf16[2560, 2560][2560, 1]cuda:0", permute_48: "bf16[2560, 2560][2560, 1]cuda:0", permute_52: "bf16[2560, 2560][2560, 1]cuda:0", permute_59: "bf16[2560, 2560][2560, 1]cuda:0", permute_63: "bf16[2560, 2560][2560, 1]cuda:0", permute_68: "bf16[2560, 2560][2560, 1]cuda:0", tangents_1: "f32[16, 128, 2560][327680, 2560, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        convert_element_type_65: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(tangents_1, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:391 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        convert_element_type_66: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_2, torch.bfloat16);  gt_2 = None
        mul_17: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_66, 1.1111111111111112);  convert_element_type_66 = None
        mul_18: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_65, mul_17);  convert_element_type_65 = mul_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_34: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(mul_18, [2048, 2560]);  mul_18 = None
        mm: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.mm.default(view_34, permute_19);  permute_19 = None
        permute_20: "bf16[2560, 2048][1, 2560]cuda:0" = torch.ops.aten.permute.default(view_34, [1, 0])
        mm_1: "bf16[2560, 10240][10240, 1]cuda:0" = torch.ops.aten.mm.default(permute_20, view_32);  permute_20 = view_32 = None
        sum_2: "f32[1, 2560][2560, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_34, [0], True, dtype = torch.float32);  view_34 = None
        view_35: "f32[2560][1]cuda:0" = torch.ops.aten.reshape.default(sum_2, [2560]);  sum_2 = None
        convert_element_type_71: "bf16[2560][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_35, torch.bfloat16);  view_35 = None
        view_36: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [16, 128, 10240]);  mm = None
        convert_element_type_72: "f32[2560, 10240][10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_73: "f32[2560][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_71, torch.float32);  convert_element_type_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_74: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_36, torch.float32);  view_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_31: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_8, [16, 128, 10240]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_58: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_31, torch.float32);  view_31 = None
        mul_13: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_58, 0.7071067811865476)
        erf: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
        add_9: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_20: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_9, 0.5);  add_9 = None
        mul_21: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_58, convert_element_type_58)
        mul_22: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_21, -0.5);  mul_21 = None
        exp_1: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.exp.default(mul_22);  mul_22 = None
        mul_23: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
        mul_24: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_58, mul_23);  convert_element_type_58 = mul_23 = None
        add_12: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_20, mul_24);  mul_20 = mul_24 = None
        mul_25: "f32[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_74, add_12);  convert_element_type_74 = add_12 = None
        convert_element_type_76: "bf16[16, 128, 10240][1310720, 10240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_25, torch.bfloat16);  mul_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_37: "bf16[2048, 10240][10240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_76, [2048, 10240]);  convert_element_type_76 = None
        mm_2: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.mm.default(view_37, permute_23);  permute_23 = None
        permute_24: "bf16[10240, 2048][1, 10240]cuda:0" = torch.ops.aten.permute.default(view_37, [1, 0])
        mm_3: "bf16[10240, 2560][2560, 1]cuda:0" = torch.ops.aten.mm.default(permute_24, view_30);  permute_24 = view_30 = None
        sum_3: "f32[1, 10240][10240, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_37, [0], True, dtype = torch.float32);  view_37 = None
        view_38: "f32[10240][1]cuda:0" = torch.ops.aten.reshape.default(sum_3, [10240]);  sum_3 = None
        convert_element_type_81: "bf16[10240][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_38, torch.bfloat16);  view_38 = None
        view_39: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(mm_2, [16, 128, 2560]);  mm_2 = None
        convert_element_type_82: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_39, torch.float32);  view_39 = None
        convert_element_type_83: "f32[10240, 2560][2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_84: "f32[10240][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_81, torch.float32);  convert_element_type_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        mul_27: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_82, primals_25);  primals_25 = None
        mul_28: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_27, 2560)
        sum_4: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_27, [2], True)
        mul_29: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_27, mul_10);  mul_27 = None
        sum_5: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_29, [2], True);  mul_29 = None
        mul_30: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, sum_5);  sum_5 = None
        sub_5: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_28, sum_4);  mul_28 = sum_4 = None
        sub_6: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_5, mul_30);  sub_5 = mul_30 = None
        mul_31: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_1, sub_6);  div_1 = sub_6 = None
        mul_32: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_82, mul_10);  mul_10 = None
        sum_6: "f32[2560][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_32, [0, 1]);  mul_32 = None
        sum_7: "f32[2560][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_82, [0, 1]);  convert_element_type_82 = None
        add_13: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(tangents_1, mul_31);  tangents_1 = mul_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        convert_element_type_85: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_13, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:382 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        convert_element_type_86: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_1, torch.bfloat16);  gt_1 = None
        mul_33: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_86, 1.1111111111111112);  convert_element_type_86 = None
        mul_34: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_85, mul_33);  convert_element_type_85 = mul_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_40: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(mul_34, [2048, 2560]);  mul_34 = None
        mm_4: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.mm.default(view_40, permute_27);  permute_27 = None
        permute_28: "bf16[2560, 2048][1, 2560]cuda:0" = torch.ops.aten.permute.default(view_40, [1, 0])
        mm_5: "bf16[2560, 2560][2560, 1]cuda:0" = torch.ops.aten.mm.default(permute_28, view_28);  permute_28 = view_28 = None
        sum_8: "f32[1, 2560][2560, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_40, [0], True, dtype = torch.float32);  view_40 = None
        view_41: "f32[2560][1]cuda:0" = torch.ops.aten.reshape.default(sum_8, [2560]);  sum_8 = None
        convert_element_type_91: "bf16[2560][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_41, torch.bfloat16);  view_41 = None
        view_42: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(mm_4, [16, 128, 2560]);  mm_4 = None
        convert_element_type_92: "f32[2560, 2560][2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_93: "f32[2560][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_91, torch.float32);  convert_element_type_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_43: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_42, [16, 128, 32, 80]);  view_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_31: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_43, [0, 2, 1, 3]);  view_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_7: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.clone.default(permute_31, memory_format = torch.contiguous_format);  permute_31 = None
        view_44: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(clone_7, [512, 128, 80]);  clone_7 = None
        expand_2: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(where_2, [16, 32, 128, 128])
        view_24: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(expand_2, [512, 128, 128]);  expand_2 = None
        permute_32: "bf16[512, 128, 128][16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_24, [0, 2, 1]);  view_24 = None
        bmm_2: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.bmm.default(permute_32, view_44);  permute_32 = None
        bmm_3: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_44, permute_33);  view_44 = permute_33 = None
        view_45: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_2, [16, 32, 128, 80]);  bmm_2 = None
        view_46: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_3, [16, 32, 128, 128]);  bmm_3 = None
        convert_element_type_98: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_46, torch.float32);  view_46 = None
        convert_element_type_99: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_2, torch.float32);  where_2 = None
        mul_35: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_98, convert_element_type_99);  convert_element_type_98 = None
        sum_9: "f32[16, 32, 128, 1][4096, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_35, [-1], True)
        neg: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_99);  convert_element_type_99 = None
        fma: "f32[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg, sum_9, mul_35);  neg = sum_9 = mul_35 = None
        convert_element_type_100: "bf16[16, 32, 128, 128][524288, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        view_47: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_100, [512, 128, 128]);  convert_element_type_100 = None
        bmm_4: "bf16[512, 80, 128][10240, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_34, view_47);  permute_34 = None
        bmm_5: "bf16[512, 128, 80][10240, 80, 1]cuda:0" = torch.ops.aten.bmm.default(view_47, permute_35);  view_47 = permute_35 = None
        view_48: "bf16[16, 32, 80, 128][327680, 10240, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_4, [16, 32, 80, 128]);  bmm_4 = None
        view_49: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_5, [16, 32, 128, 80]);  bmm_5 = None
        mul_36: "bf16[16, 32, 80, 128][327680, 10240, 128, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_48, 0.334370152488211);  view_48 = None
        permute_36: "bf16[16, 32, 128, 80][327680, 10240, 1, 128]cuda:0" = torch.ops.aten.permute.default(mul_36, [0, 1, 3, 2]);  mul_36 = None
        mul_37: "bf16[16, 32, 128, 80][327680, 10240, 80, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_49, 0.334370152488211);  view_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        permute_37: "bf16[16, 128, 32, 80][327680, 80, 10240, 1]cuda:0" = torch.ops.aten.permute.default(view_45, [0, 2, 1, 3]);  view_45 = None
        clone_8: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
        view_50: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(clone_8, [16, 128, 2560]);  clone_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        permute_38: "bf16[16, 128, 32, 80][327680, 1, 10240, 128]cuda:0" = torch.ops.aten.permute.default(permute_36, [0, 2, 1, 3]);  permute_36 = None
        view_51: "bf16[16, 128, 2560][327680, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_38, [16, 128, 2560]);  permute_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_52: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_50, [2048, 2560]);  view_50 = None
        mm_6: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.mm.default(view_52, permute_39);  permute_39 = None
        permute_40: "bf16[2560, 2048][1, 2560]cuda:0" = torch.ops.aten.permute.default(view_52, [1, 0])
        mm_7: "bf16[2560, 2560][2560, 1]cuda:0" = torch.ops.aten.mm.default(permute_40, view_15);  permute_40 = None
        sum_10: "f32[1, 2560][2560, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_52, [0], True, dtype = torch.float32);  view_52 = None
        view_53: "f32[2560][1]cuda:0" = torch.ops.aten.reshape.default(sum_10, [2560]);  sum_10 = None
        convert_element_type_109: "bf16[2560][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_53, torch.bfloat16);  view_53 = None
        view_54: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(mm_6, [16, 128, 2560]);  mm_6 = None
        convert_element_type_110: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_54, torch.float32);  view_54 = None
        convert_element_type_111: "f32[2560, 2560][2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_112: "f32[2560][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_109, torch.float32);  convert_element_type_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        clone_9: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.clone.default(view_51, memory_format = torch.contiguous_format);  view_51 = None
        view_55: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(clone_9, [2048, 2560]);  clone_9 = None
        mm_8: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.mm.default(view_55, permute_43);  permute_43 = None
        permute_44: "bf16[2560, 2048][1, 2560]cuda:0" = torch.ops.aten.permute.default(view_55, [1, 0])
        mm_9: "bf16[2560, 2560][2560, 1]cuda:0" = torch.ops.aten.mm.default(permute_44, view_15);  permute_44 = view_15 = None
        sum_11: "f32[1, 2560][2560, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_55, [0], True, dtype = torch.float32);  view_55 = None
        view_56: "f32[2560][1]cuda:0" = torch.ops.aten.reshape.default(sum_11, [2560]);  sum_11 = None
        convert_element_type_117: "bf16[2560][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_56, torch.bfloat16);  view_56 = None
        view_57: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(mm_8, [16, 128, 2560]);  mm_8 = None
        convert_element_type_118: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_57, torch.float32);  view_57 = None
        add_14: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_110, convert_element_type_118);  convert_element_type_110 = convert_element_type_118 = None
        convert_element_type_119: "f32[2560, 2560][2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        convert_element_type_120: "f32[2560][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_117, torch.float32);  convert_element_type_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_47: "bf16[16, 128, 32, 80][327680, 80, 10240, 1]cuda:0" = torch.ops.aten.permute.default(mul_37, [0, 2, 1, 3]);  mul_37 = None
        clone_10: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.clone.default(permute_47, memory_format = torch.contiguous_format);  permute_47 = None
        view_58: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(clone_10, [16, 128, 2560]);  clone_10 = None
        view_59: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_58, [2048, 2560]);  view_58 = None
        mm_10: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.mm.default(view_59, permute_48);  permute_48 = None
        permute_49: "bf16[2560, 2048][1, 2560]cuda:0" = torch.ops.aten.permute.default(view_59, [1, 0])
        mm_11: "bf16[2560, 2560][2560, 1]cuda:0" = torch.ops.aten.mm.default(permute_49, view_12);  permute_49 = view_12 = None
        sum_12: "f32[1, 2560][2560, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_59, [0], True, dtype = torch.float32);  view_59 = None
        view_60: "f32[2560][1]cuda:0" = torch.ops.aten.reshape.default(sum_12, [2560]);  sum_12 = None
        convert_element_type_125: "bf16[2560][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_60, torch.bfloat16);  view_60 = None
        view_61: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(mm_10, [16, 128, 2560]);  mm_10 = None
        convert_element_type_126: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_61, torch.float32);  view_61 = None
        convert_element_type_127: "f32[2560, 2560][2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_11, torch.float32);  mm_11 = None
        convert_element_type_128: "f32[2560][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_125, torch.float32);  convert_element_type_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        mul_39: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_126, primals_14);  primals_14 = None
        mul_40: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_39, 2560)
        sum_13: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_39, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_11: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [16, 128, 2560]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:367 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        mul_2: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt, view_11);  view_11 = None
        mul_3: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2, 1.1111111111111112);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_2: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(primals_3, mul_3);  mul_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        sub_1: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_2, getitem_12);  add_2 = getitem_12 = None
        mul_4: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_41: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_39, mul_4);  mul_39 = None
        sum_14: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_41, [2], True);  mul_41 = None
        mul_42: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4, sum_14);  sum_14 = None
        sub_8: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_40, sum_13);  mul_40 = sum_13 = None
        sub_9: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_8, mul_42);  sub_8 = mul_42 = None
        div_2: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_1, 2560);  rsqrt_1 = None
        mul_43: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_2, sub_9);  div_2 = sub_9 = None
        mul_44: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_126, mul_4);  mul_4 = None
        sum_15: "f32[2560][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_44, [0, 1]);  mul_44 = None
        sum_16: "f32[2560][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_126, [0, 1]);  convert_element_type_126 = None
        add_15: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_13, mul_43);  add_13 = mul_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        convert_element_type_129: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_15, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:367 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        convert_element_type_130: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt, torch.bfloat16);  gt = None
        mul_45: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_130, 1.1111111111111112);  convert_element_type_130 = None
        mul_46: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_129, mul_45);  convert_element_type_129 = mul_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_62: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(mul_46, [2048, 2560]);  mul_46 = None
        mm_12: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.mm.default(view_62, permute_52);  permute_52 = None
        permute_53: "bf16[2560, 2048][1, 2560]cuda:0" = torch.ops.aten.permute.default(view_62, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_6: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_9: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_6, [16, 128, -1]);  permute_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_10: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_9, [2048, 2560]);  view_9 = None
        mm_13: "bf16[2560, 2560][2560, 1]cuda:0" = torch.ops.aten.mm.default(permute_53, view_10);  permute_53 = view_10 = None
        sum_17: "f32[1, 2560][2560, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_62, [0], True, dtype = torch.float32);  view_62 = None
        view_63: "f32[2560][1]cuda:0" = torch.ops.aten.reshape.default(sum_17, [2560]);  sum_17 = None
        convert_element_type_135: "bf16[2560][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_63, torch.bfloat16);  view_63 = None
        view_64: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(mm_12, [16, 128, 2560]);  mm_12 = None
        convert_element_type_136: "f32[2560, 2560][2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_13, torch.float32);  mm_13 = None
        convert_element_type_137: "f32[2560][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_135, torch.float32);  convert_element_type_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_65: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.reshape.default(view_64, [16, 128, 32, 80]);  view_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_56: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_56, permute_1, permute_4, permute_5, getitem_2, getitem_3, getitem_8, getitem_9, where, None, None, 128, 128, 0.0, False, scale = 0.11180339887498948);  permute_56 = permute_1 = permute_4 = permute_5 = getitem_2 = getitem_3 = getitem_8 = getitem_9 = where = None
        getitem_15: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward[0]
        getitem_16: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward[1]
        getitem_17: "bf16[16, 32, 128, 80][327680, 80, 2560, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward[2];  _scaled_dot_product_cudnn_attention_backward = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        permute_57: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_17, [0, 2, 1, 3]);  getitem_17 = None
        view_66: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_57, [16, 128, 2560]);  permute_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        permute_58: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3]);  getitem_16 = None
        view_67: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_58, [16, 128, 2560]);  permute_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_68: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_66, [2048, 2560]);  view_66 = None
        mm_14: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.mm.default(view_68, permute_59);  permute_59 = None
        permute_60: "bf16[2560, 2048][1, 2560]cuda:0" = torch.ops.aten.permute.default(view_68, [1, 0])
        mm_15: "bf16[2560, 2560][2560, 1]cuda:0" = torch.ops.aten.mm.default(permute_60, view);  permute_60 = None
        sum_18: "f32[1, 2560][2560, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_68, [0], True, dtype = torch.float32);  view_68 = None
        view_69: "f32[2560][1]cuda:0" = torch.ops.aten.reshape.default(sum_18, [2560]);  sum_18 = None
        convert_element_type_142: "bf16[2560][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_69, torch.bfloat16);  view_69 = None
        view_70: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(mm_14, [16, 128, 2560]);  mm_14 = None
        convert_element_type_143: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_70, torch.float32);  view_70 = None
        convert_element_type_144: "f32[2560, 2560][2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_15, torch.float32);  mm_15 = None
        convert_element_type_145: "f32[2560][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_142, torch.float32);  convert_element_type_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_71: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_67, [2048, 2560]);  view_67 = None
        mm_16: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.mm.default(view_71, permute_63);  permute_63 = None
        permute_64: "bf16[2560, 2048][1, 2560]cuda:0" = torch.ops.aten.permute.default(view_71, [1, 0])
        mm_17: "bf16[2560, 2560][2560, 1]cuda:0" = torch.ops.aten.mm.default(permute_64, view);  permute_64 = None
        sum_19: "f32[1, 2560][2560, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_71, [0], True, dtype = torch.float32);  view_71 = None
        view_72: "f32[2560][1]cuda:0" = torch.ops.aten.reshape.default(sum_19, [2560]);  sum_19 = None
        convert_element_type_150: "bf16[2560][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_72, torch.bfloat16);  view_72 = None
        view_73: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(mm_16, [16, 128, 2560]);  mm_16 = None
        convert_element_type_151: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_73, torch.float32);  view_73 = None
        add_16: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_143, convert_element_type_151);  convert_element_type_143 = convert_element_type_151 = None
        convert_element_type_152: "f32[2560, 2560][2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_17, torch.float32);  mm_17 = None
        convert_element_type_153: "f32[2560][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_150, torch.float32);  convert_element_type_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_67: "bf16[16, 128, 32, 80][327680, 2560, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_15, [0, 2, 1, 3]);  getitem_15 = None
        view_74: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(permute_67, [16, 128, 2560]);  permute_67 = None
        view_75: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.reshape.default(view_74, [2048, 2560]);  view_74 = None
        mm_18: "bf16[2048, 2560][2560, 1]cuda:0" = torch.ops.aten.mm.default(view_75, permute_68);  permute_68 = None
        permute_69: "bf16[2560, 2048][1, 2560]cuda:0" = torch.ops.aten.permute.default(view_75, [1, 0])
        mm_19: "bf16[2560, 2560][2560, 1]cuda:0" = torch.ops.aten.mm.default(permute_69, view);  permute_69 = view = None
        sum_20: "f32[1, 2560][2560, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_75, [0], True, dtype = torch.float32);  view_75 = None
        view_76: "f32[2560][1]cuda:0" = torch.ops.aten.reshape.default(sum_20, [2560]);  sum_20 = None
        convert_element_type_158: "bf16[2560][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_76, torch.bfloat16);  view_76 = None
        view_77: "bf16[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.reshape.default(mm_18, [16, 128, 2560]);  mm_18 = None
        convert_element_type_159: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_77, torch.float32);  view_77 = None
        add_17: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_16, convert_element_type_159);  add_16 = convert_element_type_159 = None
        convert_element_type_160: "f32[2560, 2560][2560, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_19, torch.float32);  mm_19 = None
        convert_element_type_161: "f32[2560][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_158, torch.float32);  convert_element_type_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        mul_48: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_17, primals_1);  primals_1 = None
        mul_49: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_48, 2560)
        sum_21: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_48, [2], True)
        sub: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(primals_3, getitem_1);  primals_3 = getitem_1 = None
        mul: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_50: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_48, mul);  mul_48 = None
        sum_22: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_50, [2], True);  mul_50 = None
        mul_51: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, sum_22);  sum_22 = None
        sub_11: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_49, sum_21);  mul_49 = sum_21 = None
        sub_12: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_11, mul_51);  sub_11 = mul_51 = None
        div_3: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt, 2560);  rsqrt = None
        mul_52: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_3, sub_12);  div_3 = sub_12 = None
        mul_53: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_17, mul);  mul = None
        sum_23: "f32[2560][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_53, [0, 1]);  mul_53 = None
        sum_24: "f32[2560][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_17, [0, 1]);  add_17 = None
        add_18: "f32[16, 128, 2560][327680, 2560, 1]cuda:0" = torch.ops.aten.add.Tensor(add_15, mul_52);  add_15 = mul_52 = None
        return (sum_23, sum_24, add_18, convert_element_type_160, convert_element_type_161, convert_element_type_152, convert_element_type_153, convert_element_type_144, convert_element_type_145, None, convert_element_type_136, convert_element_type_137, add_14, sum_15, sum_16, convert_element_type_127, convert_element_type_128, convert_element_type_119, convert_element_type_120, convert_element_type_111, convert_element_type_112, None, convert_element_type_92, convert_element_type_93, sum_6, sum_7, convert_element_type_83, convert_element_type_84, convert_element_type_72, convert_element_type_73)

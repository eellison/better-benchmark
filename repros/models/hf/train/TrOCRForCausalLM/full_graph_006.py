class GraphModule(torch.nn.Module):
    def forward(self, primals_8: "f32[64, 1, 256, 256][65536, 65536, 256, 1]cuda:0", primals_11: "f32[1024][1]cuda:0", primals_17: "f32[1024][1]cuda:0", view: "bf16[16384, 1024][1024, 1]cuda:0", bmm: "bf16[1024, 256, 256][65536, 256, 1]cuda:0", amax: "f32[1024, 256, 1][256, 1, 1]cuda:0", sum_1: "f32[1024, 256, 1][256, 1, 1]cuda:0", view_16: "bf16[16384, 1024][1024, 1]cuda:0", gt: "b8[64, 256, 1024][262144, 1024, 1]cuda:0", mul_3: "f32[64, 256, 1024][262144, 1024, 1]cuda:0", view_18: "bf16[16384, 1024][1024, 1]cuda:0", addmm_4: "bf16[16384, 4096][4096, 1]cuda:0", view_20: "bf16[16384, 4096][4096, 1]cuda:0", gt_1: "b8[64, 256, 1024][262144, 1024, 1]cuda:0", mul_10: "f32[64, 256, 1024][262144, 1024, 1]cuda:0", div_1: "f32[64, 256, 1][256, 1, 1]cuda:0", permute_11: "bf16[1024, 4096][4096, 1]cuda:0", permute_15: "bf16[4096, 1024][1024, 1]cuda:0", div_2: "f32[64, 256, 1][256, 1, 1]cuda:0", permute_19: "bf16[1024, 1024][1024, 1]cuda:0", permute_24: "bf16[1024, 256, 256][65536, 1, 256]cuda:0", permute_25: "bf16[1024, 64, 256][16384, 1, 64]cuda:0", permute_26: "bf16[1024, 64, 256][16384, 1, 64]cuda:0", permute_27: "bf16[1024, 256, 64][16384, 64, 1]cuda:0", permute_32: "bf16[1024, 1024][1024, 1]cuda:0", permute_36: "bf16[1024, 1024][1024, 1]cuda:0", permute_40: "bf16[1024, 1024][1024, 1]cuda:0", tangents_1: "f32[64, 256, 1024][262144, 1024, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:380 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        mul_13: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(tangents_1, primals_17);  primals_17 = None
        mul_14: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_13, 1024)
        sum_2: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_13, [2], True)
        mul_15: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_13, mul_10);  mul_13 = None
        sum_3: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_15, [2], True);  mul_15 = None
        mul_16: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, sum_3);  sum_3 = None
        sub_4: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_14, sum_2);  mul_14 = sum_2 = None
        sub_5: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_4, mul_16);  sub_4 = mul_16 = None
        mul_17: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_1, sub_5);  div_1 = sub_5 = None
        mul_18: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(tangents_1, mul_10);  mul_10 = None
        sum_4: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_18, [0, 1]);  mul_18 = None
        sum_5: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(tangents_1, [0, 1]);  tangents_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:379 in forward, code: hidden_states = residual + hidden_states
        convert_element_type_41: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_17, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:378 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        convert_element_type_42: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_1, torch.bfloat16);  gt_1 = None
        mul_19: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_42, 1.1111111111111112);  convert_element_type_42 = None
        mul_20: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_41, mul_19);  convert_element_type_41 = mul_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:376 in forward, code: hidden_states = self.fc2(hidden_states)
        view_22: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_20, [16384, 1024]);  mul_20 = None
        mm: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_22, permute_11);  permute_11 = None
        permute_12: "bf16[1024, 16384][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_22, [1, 0])
        mm_1: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_12, view_20);  permute_12 = view_20 = None
        sum_6: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_22, [0], True, dtype = torch.float32);  view_22 = None
        view_23: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_6, [1024]);  sum_6 = None
        convert_element_type_47: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_23, torch.bfloat16);  view_23 = None
        view_24: "bf16[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [64, 256, 4096]);  mm = None
        convert_element_type_48: "f32[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_49: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_47, torch.float32);  convert_element_type_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_50: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_24, torch.float32);  view_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:374 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_19: "bf16[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [64, 256, 4096]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_34: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_19, torch.float32);  view_19 = None
        mul_6: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_34, 0.7071067811865476)
        erf: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
        add_4: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_22: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_4, 0.5);  add_4 = None
        mul_23: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_34, convert_element_type_34)
        mul_24: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_23, -0.5);  mul_23 = None
        exp_1: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.exp.default(mul_24);  mul_24 = None
        mul_25: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
        mul_26: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_34, mul_25);  convert_element_type_34 = mul_25 = None
        add_9: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_22, mul_26);  mul_22 = mul_26 = None
        mul_27: "f32[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_50, add_9);  convert_element_type_50 = add_9 = None
        convert_element_type_52: "bf16[64, 256, 4096][1048576, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_27, torch.bfloat16);  mul_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:374 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_25: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_52, [16384, 4096]);  convert_element_type_52 = None
        mm_2: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_25, permute_15);  permute_15 = None
        permute_16: "bf16[4096, 16384][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_25, [1, 0])
        mm_3: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_16, view_18);  permute_16 = view_18 = None
        sum_7: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_25, [0], True, dtype = torch.float32);  view_25 = None
        view_26: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_7, [4096]);  sum_7 = None
        convert_element_type_57: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_26, torch.bfloat16);  view_26 = None
        view_27: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_2, [64, 256, 1024]);  mm_2 = None
        convert_element_type_58: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_27, torch.float32);  view_27 = None
        add_10: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_17, convert_element_type_58);  mul_17 = convert_element_type_58 = None
        convert_element_type_59: "f32[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_60: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_57, torch.float32);  convert_element_type_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:353 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        mul_29: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_10, primals_11);  primals_11 = None
        mul_30: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_29, 1024)
        sum_8: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_29, [2], True)
        mul_31: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_29, mul_3);  mul_29 = None
        sum_9: "f32[64, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_31, [2], True);  mul_31 = None
        mul_32: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3, sum_9);  sum_9 = None
        sub_7: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_30, sum_8);  mul_30 = sum_8 = None
        sub_8: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_7, mul_32);  sub_7 = mul_32 = None
        mul_33: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_2, sub_8);  div_2 = sub_8 = None
        mul_34: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_10, mul_3);  mul_3 = None
        sum_10: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_34, [0, 1]);  mul_34 = None
        sum_11: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_10, [0, 1]);  add_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:352 in forward, code: hidden_states = residual + hidden_states
        convert_element_type_61: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_33, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:351 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        convert_element_type_62: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt, torch.bfloat16);  gt = None
        mul_35: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_62, 1.1111111111111112);  convert_element_type_62 = None
        mul_36: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_61, mul_35);  convert_element_type_61 = mul_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:274 in forward, code: attn_output = self.out_proj(attn_output)
        view_28: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_36, [16384, 1024]);  mul_36 = None
        mm_4: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_28, permute_19);  permute_19 = None
        permute_20: "bf16[1024, 16384][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_28, [1, 0])
        mm_5: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_20, view_16);  permute_20 = view_16 = None
        sum_12: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_28, [0], True, dtype = torch.float32);  view_28 = None
        view_29: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_12, [1024]);  sum_12 = None
        convert_element_type_67: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_29, torch.bfloat16);  view_29 = None
        view_30: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_4, [64, 256, 1024]);  mm_4 = None
        convert_element_type_68: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_69: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_67, torch.float32);  convert_element_type_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:272 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
        view_31: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_30, [64, 256, 16, 64]);  view_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:271 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_23: "bf16[64, 16, 256, 64][262144, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:270 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        clone_8: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
        view_32: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_8, [1024, 256, 64]);  clone_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:262 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_2: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_24, view_32);  permute_24 = None
        bmm_3: "bf16[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_32, permute_25);  view_32 = permute_25 = None
        convert_element_type_74: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(bmm_3, torch.float32);  bmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:245 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_12: "bf16[64, 16, 256, 256][1048576, 65536, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm, [64, 16, 256, 256]);  bmm = None
        add: "f32[64, 16, 256, 256][1048576, 65536, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(view_12, primals_8);  view_12 = primals_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:246 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_13: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.reshape.default(add, [1024, 256, 256]);  add = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:248 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        sub: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_13, amax);  view_13 = amax = None
        exp: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.exp.default(sub);  sub = None
        div: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        mul_37: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_74, div);  convert_element_type_74 = None
        sum_13: "f32[1024, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_37, [-1], True)
        neg: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.neg.default(div);  div = None
        fma: "f32[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.prims.fma.default(neg, sum_13, mul_37);  neg = sum_13 = mul_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:246 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_33: "f32[64, 16, 256, 256][1048576, 65536, 256, 1]cuda:0" = torch.ops.aten.reshape.default(fma, [64, 16, 256, 256]);  fma = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:245 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        convert_element_type_75: "bf16[64, 16, 256, 256][1048576, 65536, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_33, torch.bfloat16);  view_33 = None
        view_34: "bf16[1024, 256, 256][65536, 256, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_75, [1024, 256, 256]);  convert_element_type_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:232 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        bmm_4: "bf16[1024, 64, 256][16384, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_26, view_34);  permute_26 = None
        bmm_5: "bf16[1024, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_34, permute_27);  view_34 = permute_27 = None
        permute_28: "bf16[1024, 256, 64][16384, 1, 256]cuda:0" = torch.ops.aten.permute.default(bmm_4, [0, 2, 1]);  bmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:229 in forward, code: value_states = value_states.reshape(*proj_shape)
        view_35: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_2, [64, 16, 256, 64]);  bmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:228 in forward, code: key_states = key_states.reshape(*proj_shape)
        view_36: "bf16[64, 16, 256, 64][262144, 16384, 1, 256]cuda:0" = torch.ops.aten.reshape.default(permute_28, [64, 16, 256, 64]);  permute_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:227 in forward, code: query_states = query_states.reshape(*proj_shape)
        view_37: "bf16[64, 16, 256, 64][262144, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_5, [64, 16, 256, 64]);  bmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:226 in forward, code: query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        permute_29: "bf16[64, 256, 16, 64][262144, 64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_37, [0, 2, 1, 3]);  view_37 = None
        clone_9: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_38: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_9, [64, 256, 1024]);  clone_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:216 in forward, code: value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        permute_30: "bf16[64, 256, 16, 64][262144, 64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_35, [0, 2, 1, 3]);  view_35 = None
        clone_10: "bf16[64, 256, 16, 64][262144, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
        view_39: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_10, [64, 256, 1024]);  clone_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:215 in forward, code: key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        permute_31: "bf16[64, 256, 16, 64][262144, 1, 16384, 256]cuda:0" = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None
        view_40: "bf16[64, 256, 1024][262144, 1, 256]cuda:0" = torch.ops.aten.reshape.default(permute_31, [64, 256, 1024]);  permute_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_41: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_39, [16384, 1024]);  view_39 = None
        mm_6: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_41, permute_32);  permute_32 = None
        permute_33: "bf16[1024, 16384][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_41, [1, 0])
        mm_7: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_33, view);  permute_33 = None
        sum_14: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_41, [0], True, dtype = torch.float32);  view_41 = None
        view_42: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_14, [1024]);  sum_14 = None
        convert_element_type_84: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_42, torch.bfloat16);  view_42 = None
        view_43: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_6, [64, 256, 1024]);  mm_6 = None
        convert_element_type_85: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_43, torch.float32);  view_43 = None
        add_11: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_33, convert_element_type_85);  mul_33 = convert_element_type_85 = None
        convert_element_type_86: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_87: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_84, torch.float32);  convert_element_type_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:213 in forward, code: key_states = self.k_proj(current_states)
        clone_11: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.clone.default(view_40, memory_format = torch.contiguous_format);  view_40 = None
        view_44: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_11, [16384, 1024]);  clone_11 = None
        mm_8: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_44, permute_36);  permute_36 = None
        permute_37: "bf16[1024, 16384][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_44, [1, 0])
        mm_9: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_37, view);  permute_37 = None
        sum_15: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_44, [0], True, dtype = torch.float32);  view_44 = None
        view_45: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_15, [1024]);  sum_15 = None
        convert_element_type_92: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_45, torch.bfloat16);  view_45 = None
        view_46: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_8, [64, 256, 1024]);  mm_8 = None
        convert_element_type_93: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_46, torch.float32);  view_46 = None
        add_12: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_11, convert_element_type_93);  add_11 = convert_element_type_93 = None
        convert_element_type_94: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        convert_element_type_95: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_92, torch.float32);  convert_element_type_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:193 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        mul_38: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_38, 0.125);  view_38 = None
        view_47: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_38, [16384, 1024]);  mul_38 = None
        mm_10: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_47, permute_40);  permute_40 = None
        permute_41: "bf16[1024, 16384][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_47, [1, 0])
        mm_11: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_41, view);  permute_41 = view = None
        sum_16: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_47, [0], True, dtype = torch.float32);  view_47 = None
        view_48: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_16, [1024]);  sum_16 = None
        convert_element_type_100: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_48, torch.bfloat16);  view_48 = None
        view_49: "bf16[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_10, [64, 256, 1024]);  mm_10 = None
        convert_element_type_101: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_49, torch.float32);  view_49 = None
        add_13: "f32[64, 256, 1024][262144, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_12, convert_element_type_101);  add_12 = convert_element_type_101 = None
        convert_element_type_102: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_11, torch.float32);  mm_11 = None
        convert_element_type_103: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_100, torch.float32);  convert_element_type_100 = None
        return (add_13, convert_element_type_102, convert_element_type_103, convert_element_type_94, convert_element_type_95, convert_element_type_86, convert_element_type_87, None, convert_element_type_68, convert_element_type_69, sum_10, sum_11, convert_element_type_59, convert_element_type_60, convert_element_type_48, convert_element_type_49, sum_4, sum_5)

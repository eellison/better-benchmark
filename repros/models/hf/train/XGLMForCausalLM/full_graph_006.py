class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[1024][1]cuda:0", primals_3: "f32[32, 128, 1024][131072, 1024, 1]cuda:0", primals_10: "f32[32, 1, 128, 128][16384, 16384, 128, 1]cuda:0", primals_13: "f32[1024][1]cuda:0", getitem_1: "f32[32, 128, 1][128, 1, 1]cuda:0", rsqrt: "f32[32, 128, 1][128, 1, 1]cuda:0", view: "bf16[4096, 1024][1024, 1]cuda:0", bmm: "bf16[512, 128, 128][16384, 128, 1]cuda:0", amax: "f32[512, 128, 1][128, 1, 1]cuda:0", sum_1: "f32[512, 128, 1][128, 1, 1]cuda:0", gt: "b8[512, 128, 128][16384, 128, 1]cuda:0", view_18: "bf16[4096, 1024][1024, 1]cuda:0", addmm_3: "bf16[4096, 1024][1024, 1]cuda:0", gt_1: "b8[32, 128, 1024][131072, 1024, 1]cuda:0", getitem_3: "f32[32, 128, 1][128, 1, 1]cuda:0", rsqrt_1: "f32[32, 128, 1][128, 1, 1]cuda:0", view_20: "bf16[4096, 1024][1024, 1]cuda:0", addmm_4: "bf16[4096, 4096][4096, 1]cuda:0", view_22: "bf16[4096, 4096][4096, 1]cuda:0", gt_2: "b8[32, 128, 1024][131072, 1024, 1]cuda:0", permute_11: "bf16[1024, 4096][4096, 1]cuda:0", permute_15: "bf16[4096, 1024][1024, 1]cuda:0", permute_19: "bf16[1024, 1024][1024, 1]cuda:0", permute_24: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_25: "bf16[512, 64, 128][8192, 1, 64]cuda:0", permute_26: "bf16[512, 64, 128][8192, 1, 64]cuda:0", permute_27: "bf16[512, 128, 64][8192, 64, 1]cuda:0", permute_32: "bf16[1024, 1024][1024, 1]cuda:0", permute_36: "bf16[1024, 1024][1024, 1]cuda:0", permute_40: "bf16[1024, 1024][1024, 1]cuda:0", tangents_1: "f32[32, 128, 1024][131072, 1024, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:336 in forward, code: hidden_states = residual + hidden_states
        convert_element_type_41: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(tangents_1, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:335 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        convert_element_type_42: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_2, torch.bfloat16);  gt_2 = None
        mul_14: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_42, 1.1111111111111112);  convert_element_type_42 = None
        mul_15: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_41, mul_14);  convert_element_type_41 = mul_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:334 in forward, code: hidden_states = self.fc2(hidden_states)
        view_24: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_15, [4096, 1024]);  mul_15 = None
        mm: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_24, permute_11);  permute_11 = None
        permute_12: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_24, [1, 0])
        mm_1: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_12, view_22);  permute_12 = view_22 = None
        sum_2: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_24, [0], True, dtype = torch.float32);  view_24 = None
        view_25: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_2, [1024]);  sum_2 = None
        convert_element_type_47: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_25, torch.bfloat16);  view_25 = None
        view_26: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [32, 128, 4096]);  mm = None
        convert_element_type_48: "f32[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_49: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_47, torch.float32);  convert_element_type_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_50: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_26, torch.float32);  view_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:332 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_21: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [32, 128, 4096]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_34: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_21, torch.float32);  view_21 = None
        mul_10: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_34, 0.7071067811865476)
        erf: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_10);  mul_10 = None
        add_6: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_17: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_6, 0.5);  add_6 = None
        mul_18: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_34, convert_element_type_34)
        mul_19: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_18, -0.5);  mul_18 = None
        exp_1: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.exp.default(mul_19);  mul_19 = None
        mul_20: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
        mul_21: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_34, mul_20);  convert_element_type_34 = mul_20 = None
        add_9: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_17, mul_21);  mul_17 = mul_21 = None
        mul_22: "f32[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_50, add_9);  convert_element_type_50 = add_9 = None
        convert_element_type_52: "bf16[32, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_22, torch.bfloat16);  mul_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:332 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_27: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_52, [4096, 4096]);  convert_element_type_52 = None
        mm_2: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_27, permute_15);  permute_15 = None
        permute_16: "bf16[4096, 4096][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_27, [1, 0])
        mm_3: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_16, view_20);  permute_16 = view_20 = None
        sum_3: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_27, [0], True, dtype = torch.float32);  view_27 = None
        view_28: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_3, [4096]);  sum_3 = None
        convert_element_type_57: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_28, torch.bfloat16);  view_28 = None
        view_29: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_2, [32, 128, 1024]);  mm_2 = None
        convert_element_type_58: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_29, torch.float32);  view_29 = None
        convert_element_type_59: "f32[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_60: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_57, torch.float32);  convert_element_type_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:331 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        mul_24: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_58, primals_13);  primals_13 = None
        mul_25: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_24, 1024)
        sum_4: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_24, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:243 in forward, code: attn_output = self.out_proj(attn_output)
        view_19: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [32, 128, 1024]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:311 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        mul_5: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_1, view_19);  view_19 = None
        mul_6: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_5, 1.1111111111111112);  mul_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:312 in forward, code: hidden_states = residual + hidden_states
        add_3: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(primals_3, mul_6);  mul_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:331 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        sub_2: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_3, getitem_3);  add_3 = getitem_3 = None
        mul_7: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
        mul_26: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_24, mul_7);  mul_24 = None
        sum_5: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_26, [2], True);  mul_26 = None
        mul_27: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_7, sum_5);  sum_5 = None
        sub_4: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_25, sum_4);  mul_25 = sum_4 = None
        sub_5: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_4, mul_27);  sub_4 = mul_27 = None
        div_1: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_1, 1024);  rsqrt_1 = None
        mul_28: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_1, sub_5);  div_1 = sub_5 = None
        mul_29: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_58, mul_7);  mul_7 = None
        sum_6: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_29, [0, 1]);  mul_29 = None
        sum_7: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_58, [0, 1]);  convert_element_type_58 = None
        add_10: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(tangents_1, mul_28);  tangents_1 = mul_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:312 in forward, code: hidden_states = residual + hidden_states
        convert_element_type_61: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_10, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:311 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        convert_element_type_62: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_1, torch.bfloat16);  gt_1 = None
        mul_30: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_62, 1.1111111111111112);  convert_element_type_62 = None
        mul_31: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_61, mul_30);  convert_element_type_61 = mul_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:243 in forward, code: attn_output = self.out_proj(attn_output)
        view_30: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_31, [4096, 1024]);  mul_31 = None
        mm_4: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_30, permute_19);  permute_19 = None
        permute_20: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_30, [1, 0])
        mm_5: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_20, view_18);  permute_20 = view_18 = None
        sum_8: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_30, [0], True, dtype = torch.float32);  view_30 = None
        view_31: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_8, [1024]);  sum_8 = None
        convert_element_type_67: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_31, torch.bfloat16);  view_31 = None
        view_32: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_4, [32, 128, 1024]);  mm_4 = None
        convert_element_type_68: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_69: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_67, torch.float32);  convert_element_type_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:241 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        view_33: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_32, [32, 128, 16, 64]);  view_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:237 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_23: "bf16[32, 16, 128, 64][131072, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3]);  view_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:236 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        clone_7: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
        view_34: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_7, [512, 128, 64]);  clone_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:228 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_2: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_24, view_34);  permute_24 = None
        bmm_3: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_34, permute_25);  view_34 = permute_25 = None
        convert_element_type_74: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(bmm_3, torch.float32);  bmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:226 in forward, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_75: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_32: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_75, 1.1111111111111112);  convert_element_type_75 = None
        mul_33: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_74, mul_32);  convert_element_type_74 = mul_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:207 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_12: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm, [32, 16, 128, 128]);  bmm = None
        add_2: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_12, primals_10);  view_12 = primals_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:209 in forward, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        full_default: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:208 in forward, code: attn_weights = torch.max(
        maximum: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.maximum.default(add_2, full_default)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:211 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_13: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(maximum, [512, 128, 128]);  maximum = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:217 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        sub_1: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_13, amax);  view_13 = amax = None
        exp: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        div: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        mul_34: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_33, div);  mul_33 = None
        sum_9: "f32[512, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_34, [-1], True)
        neg: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div);  div = None
        fma: "f32[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg, sum_9, mul_34);  neg = sum_9 = mul_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:211 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_37: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(fma, [32, 16, 128, 128]);  fma = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:208 in forward, code: attn_weights = torch.max(
        div_2: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Scalar(view_37, 2)
        eq: "b8[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Tensor(add_2, full_default)
        where: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(eq, div_2, view_37);  eq = div_2 = view_37 = None
        lt: "b8[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.lt.Tensor(add_2, full_default);  add_2 = full_default = None
        full_default_1: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(lt, full_default_1, where);  lt = full_default_1 = where = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:207 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        convert_element_type_76: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_1, torch.bfloat16);  where_1 = None
        view_38: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_76, [512, 128, 128]);  convert_element_type_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:194 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        bmm_4: "bf16[512, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_26, view_38);  permute_26 = None
        bmm_5: "bf16[512, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_38, permute_27);  view_38 = permute_27 = None
        permute_28: "bf16[512, 128, 64][8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(bmm_4, [0, 2, 1]);  bmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:191 in forward, code: value_states = value_states.reshape(*proj_shape)
        view_39: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_2, [32, 16, 128, 64]);  bmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:190 in forward, code: key_states = key_states.reshape(*proj_shape)
        view_40: "bf16[32, 16, 128, 64][131072, 8192, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_28, [32, 16, 128, 64]);  permute_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:189 in forward, code: query_states = query_states.reshape(*proj_shape)
        view_41: "bf16[32, 16, 128, 64][131072, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_5, [32, 16, 128, 64]);  bmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:188 in forward, code: query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        permute_29: "bf16[32, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None
        clone_9: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_42: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_9, [32, 128, 1024]);  clone_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:178 in forward, code: value_states = value_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        permute_30: "bf16[32, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_39, [0, 2, 1, 3]);  view_39 = None
        clone_10: "bf16[32, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
        view_43: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_10, [32, 128, 1024]);  clone_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:177 in forward, code: key_states = key_states.view(bsz, src_len, -1, self.head_dim).transpose(1, 2)
        permute_31: "bf16[32, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
        view_44: "bf16[32, 128, 1024][131072, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_31, [32, 128, 1024]);  permute_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:176 in forward, code: value_states = self.v_proj(current_states)
        view_45: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_43, [4096, 1024]);  view_43 = None
        mm_6: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_45, permute_32);  permute_32 = None
        permute_33: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_45, [1, 0])
        mm_7: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_33, view);  permute_33 = None
        sum_10: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_45, [0], True, dtype = torch.float32);  view_45 = None
        view_46: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_10, [1024]);  sum_10 = None
        convert_element_type_85: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_46, torch.bfloat16);  view_46 = None
        view_47: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_6, [32, 128, 1024]);  mm_6 = None
        convert_element_type_86: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_47, torch.float32);  view_47 = None
        convert_element_type_87: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_88: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_85, torch.float32);  convert_element_type_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:175 in forward, code: key_states = self.k_proj(current_states)
        clone_11: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.clone.default(view_44, memory_format = torch.contiguous_format);  view_44 = None
        view_48: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_11, [4096, 1024]);  clone_11 = None
        mm_8: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_48, permute_36);  permute_36 = None
        permute_37: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_48, [1, 0])
        mm_9: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_37, view);  permute_37 = None
        sum_11: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_48, [0], True, dtype = torch.float32);  view_48 = None
        view_49: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_11, [1024]);  sum_11 = None
        convert_element_type_93: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_49, torch.bfloat16);  view_49 = None
        view_50: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_8, [32, 128, 1024]);  mm_8 = None
        convert_element_type_94: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_50, torch.float32);  view_50 = None
        add_11: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_86, convert_element_type_94);  convert_element_type_86 = convert_element_type_94 = None
        convert_element_type_95: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        convert_element_type_96: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_93, torch.float32);  convert_element_type_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:155 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        mul_35: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_42, 0.125);  view_42 = None
        view_51: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_35, [4096, 1024]);  mul_35 = None
        mm_10: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_51, permute_40);  permute_40 = None
        permute_41: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_51, [1, 0])
        mm_11: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_41, view);  permute_41 = view = None
        sum_12: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_51, [0], True, dtype = torch.float32);  view_51 = None
        view_52: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_12, [1024]);  sum_12 = None
        convert_element_type_101: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_52, torch.bfloat16);  view_52 = None
        view_53: "bf16[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_10, [32, 128, 1024]);  mm_10 = None
        convert_element_type_102: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_53, torch.float32);  view_53 = None
        add_12: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_11, convert_element_type_102);  add_11 = convert_element_type_102 = None
        convert_element_type_103: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_11, torch.float32);  mm_11 = None
        convert_element_type_104: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_101, torch.float32);  convert_element_type_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:302 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        mul_37: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_12, primals_1);  primals_1 = None
        mul_38: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_37, 1024)
        sum_13: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_37, [2], True)
        sub: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(primals_3, getitem_1);  primals_3 = getitem_1 = None
        mul: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_39: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_37, mul);  mul_37 = None
        sum_14: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_39, [2], True);  mul_39 = None
        mul_40: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, sum_14);  sum_14 = None
        sub_7: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_38, sum_13);  mul_38 = sum_13 = None
        sub_8: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_7, mul_40);  sub_7 = mul_40 = None
        div_3: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt, 1024);  rsqrt = None
        mul_41: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_3, sub_8);  div_3 = sub_8 = None
        mul_42: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_12, mul);  mul = None
        sum_15: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_42, [0, 1]);  mul_42 = None
        sum_16: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_12, [0, 1]);  add_12 = None
        add_13: "f32[32, 128, 1024][131072, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_10, mul_41);  add_10 = mul_41 = None
        return (sum_15, sum_16, add_13, convert_element_type_103, convert_element_type_104, convert_element_type_95, convert_element_type_96, convert_element_type_87, convert_element_type_88, None, convert_element_type_68, convert_element_type_69, sum_6, sum_7, convert_element_type_59, convert_element_type_60, convert_element_type_48, convert_element_type_49)

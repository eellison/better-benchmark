import torch
import torch._inductor.inductor_prims  # registers inductor prims ops
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[4, 512, 768]", primals_2: "f32[768, 768]", primals_4: "f32[768, 768]", primals_6: "f32[768, 768]", primals_8: "b8[4, 1, 512, 512]", primals_9: "f32[768, 768]", primals_11: "f32[768]", primals_13: "f32[3072, 768]", primals_15: "f32[768, 3072]", primals_17: "f32[768]", bmm: "f32[48, 512, 512]", amax: "f32[4, 12, 512, 1]", sum_1: "f32[4, 12, 512, 1]", logical_not_1: "b8[4, 12, 512, 1]", gt: "b8[4, 12, 512, 512]", view_16: "f32[2048, 768]", addmm_3: "f32[2048, 768]", gt_1: "b8[4, 512, 768]", getitem_1: "f32[4, 512, 1]", rsqrt: "f32[4, 512, 1]", view_18: "f32[2048, 768]", addmm_4: "f32[2048, 3072]", gt_2: "b8[4, 512, 3072]", view_20: "f32[2048, 3072]", gt_3: "b8[4, 512, 768]", mul_15: "f32[4, 512, 768]", div_1: "f32[4, 512, 1]", permute_24: "f32[48, 512, 512]", permute_25: "f32[48, 64, 512]", permute_26: "f32[48, 64, 512]", permute_27: "f32[48, 512, 64]", tangents_1: "f32[4, 512, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:302 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        mul_18: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(tangents_1, primals_17);  primals_17 = None
        mul_19: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_18, 768)
        sum_2: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_18, [2], True)
        mul_20: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_18, mul_15);  mul_18 = None
        sum_3: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_20, [2], True);  mul_20 = None
        mul_21: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_15, sum_3);  sum_3 = None
        sub_4: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_19, sum_2);  mul_19 = sum_2 = None
        sub_5: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_4, mul_21);  sub_4 = mul_21 = None
        mul_22: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_1, sub_5);  div_1 = sub_5 = None
        mul_23: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(tangents_1, mul_15);  mul_15 = None
        sum_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_23, [0, 1]);  mul_23 = None
        sum_5: "f32[768]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0, 1]);  tangents_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:300 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        convert_element_type: "f32[4, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_3, torch.float32);  gt_3 = None
        mul_24: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.1111111111111112);  convert_element_type = None
        mul_25: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_22, mul_24);  mul_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:299 in forward, code: hidden_states = self.fc2(hidden_states)
        view_22: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_25, [2048, 768]);  mul_25 = None
        permute_10: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
        permute_11: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        mm: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_22, permute_11);  permute_11 = None
        permute_12: "f32[768, 2048]" = torch.ops.aten.permute.default(view_22, [1, 0])
        mm_1: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_12, view_20);  permute_12 = view_20 = None
        sum_6: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_22, [0], True);  view_22 = None
        view_23: "f32[768]" = torch.ops.aten.reshape.default(sum_6, [768]);  sum_6 = None
        view_24: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm, [4, 512, 3072]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:298 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        convert_element_type_1: "f32[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(gt_2, torch.float32);  gt_2 = None
        mul_26: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_27: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_24, mul_26);  view_24 = mul_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:297 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_19: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_4, [4, 512, 3072]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_9: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476)
        erf: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_9);  mul_9 = None
        add_4: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_29: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_4, 0.5);  add_4 = None
        mul_30: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, view_19)
        mul_31: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_30, -0.5);  mul_30 = None
        exp_1: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_31);  mul_31 = None
        mul_32: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
        mul_33: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, mul_32);  view_19 = mul_32 = None
        add_9: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_29, mul_33);  mul_29 = mul_33 = None
        mul_34: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_27, add_9);  mul_27 = add_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:297 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_25: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_34, [2048, 3072]);  mul_34 = None
        permute_9: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
        permute_15: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        mm_2: "f32[2048, 768]" = torch.ops.aten.mm.default(view_25, permute_15);  permute_15 = None
        permute_16: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_25, [1, 0])
        mm_3: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_16, view_18);  permute_16 = view_18 = None
        sum_7: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_25, [0], True);  view_25 = None
        view_26: "f32[3072]" = torch.ops.aten.reshape.default(sum_7, [3072]);  sum_7 = None
        view_27: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_2, [4, 512, 768]);  mm_2 = None
        add_10: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_22, view_27);  mul_22 = view_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:294 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        mul_36: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_10, primals_11);  primals_11 = None
        mul_37: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_36, 768)
        sum_8: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_36, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:255 in forward, code: attn_output = self.out_proj(attn_output)
        view_17: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_3, [4, 512, 768]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:292 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        mul_4: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(gt_1, view_17);  view_17 = None
        mul_5: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_4, 1.1111111111111112);  mul_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:293 in forward, code: hidden_states = residual + hidden_states
        add_1: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(primals_1, mul_5);  mul_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:294 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        sub_1: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
        mul_6: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = None
        mul_38: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_36, mul_6);  mul_36 = None
        sum_9: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_38, [2], True);  mul_38 = None
        mul_39: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_6, sum_9);  sum_9 = None
        sub_7: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_37, sum_8);  mul_37 = sum_8 = None
        sub_8: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_7, mul_39);  sub_7 = mul_39 = None
        div_2: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        mul_40: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_2, sub_8);  div_2 = sub_8 = None
        mul_41: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_10, mul_6);  mul_6 = None
        sum_10: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_41, [0, 1]);  mul_41 = None
        sum_11: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_10, [0, 1]);  add_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:292 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        convert_element_type_2: "f32[4, 512, 768]" = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_42: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
        mul_43: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_40, mul_42);  mul_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:255 in forward, code: attn_output = self.out_proj(attn_output)
        view_28: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_43, [2048, 768]);  mul_43 = None
        permute_8: "f32[768, 768]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
        permute_19: "f32[768, 768]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        mm_4: "f32[2048, 768]" = torch.ops.aten.mm.default(view_28, permute_19);  permute_19 = None
        permute_20: "f32[768, 2048]" = torch.ops.aten.permute.default(view_28, [1, 0])
        mm_5: "f32[768, 768]" = torch.ops.aten.mm.default(permute_20, view_16);  permute_20 = view_16 = None
        sum_12: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_28, [0], True);  view_28 = None
        view_29: "f32[768]" = torch.ops.aten.reshape.default(sum_12, [768]);  sum_12 = None
        view_30: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_4, [4, 512, 768]);  mm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:254 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_31: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_30, [4, 512, 12, 64]);  view_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_23: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_7: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
        view_32: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_7, [48, 512, 64]);  clone_7 = None
        bmm_2: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_24, view_32);  permute_24 = None
        bmm_3: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_32, permute_25);  view_32 = permute_25 = None
        view_33: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_2, [4, 12, 512, 64]);  bmm_2 = None
        view_34: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_3, [4, 12, 512, 512]);  bmm_3 = None
        convert_element_type_3: "f32[4, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_44: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.1111111111111112);  convert_element_type_3 = None
        mul_45: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_34, mul_44);  view_34 = mul_44 = None
        full_default: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[4, 1, 512, 512]" = torch.ops.aten.where.self(primals_8, full_default_1, full_default);  primals_8 = full_default_1 = full_default = None
        view_11: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm, [4, 12, 512, 512]);  bmm = None
        add: "f32[4, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_11, where);  view_11 = where = None
        sub: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add, amax);  add = amax = None
        exp: "f32[4, 12, 512, 512]" = torch.ops.aten.exp.default(sub);  sub = None
        div: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        full_default_2: "f32[4, 12, 512, 512]" = torch.ops.aten.full.default([4, 12, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[4, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_1, full_default_2, div);  logical_not_1 = full_default_2 = div = None
        mul_46: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_45, where_1);  mul_45 = None
        sum_13: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_46, [-1], True)
        neg: "f32[4, 12, 512, 512]" = torch.ops.aten.neg.default(where_1);  where_1 = None
        fma: "f32[4, 12, 512, 512]" = torch.ops.prims.fma.default(neg, sum_13, mul_46);  neg = sum_13 = mul_46 = None
        view_35: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(fma, [48, 512, 512]);  fma = None
        bmm_4: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_26, view_35);  permute_26 = None
        bmm_5: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_35, permute_27);  view_35 = permute_27 = None
        view_36: "f32[4, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_4, [4, 12, 64, 512]);  bmm_4 = None
        view_37: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_5, [4, 12, 512, 64]);  bmm_5 = None
        mul_47: "f32[4, 12, 64, 512]" = torch.ops.aten.mul.Scalar(view_36, 0.3535533905932738);  view_36 = None
        permute_28: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(mul_47, [0, 1, 3, 2]);  mul_47 = None
        mul_48: "f32[4, 12, 512, 64]" = torch.ops.aten.mul.Scalar(view_37, 0.3535533905932738);  view_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:231 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        permute_29: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3]);  view_33 = None
        clone_9: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_38: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_9, [4, 512, 768]);  clone_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:230 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        permute_30: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(permute_28, [0, 2, 1, 3]);  permute_28 = None
        view_39: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(permute_30, [4, 512, 768]);  permute_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:228 in forward, code: value_states = self.v_proj(current_states)
        view_40: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_38, [2048, 768]);  view_38 = None
        permute_3: "f32[768, 768]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        permute_31: "f32[768, 768]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        mm_6: "f32[2048, 768]" = torch.ops.aten.mm.default(view_40, permute_31);  permute_31 = None
        permute_32: "f32[768, 2048]" = torch.ops.aten.permute.default(view_40, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:207 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view: "f32[2048, 768]" = torch.ops.aten.reshape.default(primals_1, [2048, 768]);  primals_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:228 in forward, code: value_states = self.v_proj(current_states)
        mm_7: "f32[768, 768]" = torch.ops.aten.mm.default(permute_32, view);  permute_32 = None
        sum_14: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_40, [0], True);  view_40 = None
        view_41: "f32[768]" = torch.ops.aten.reshape.default(sum_14, [768]);  sum_14 = None
        view_42: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_6, [4, 512, 768]);  mm_6 = None
        add_11: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_40, view_42);  mul_40 = view_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:227 in forward, code: key_states = self.k_proj(current_states)
        clone_10: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_39, memory_format = torch.contiguous_format);  view_39 = None
        view_43: "f32[2048, 768]" = torch.ops.aten.reshape.default(clone_10, [2048, 768]);  clone_10 = None
        permute_2: "f32[768, 768]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        permute_35: "f32[768, 768]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_8: "f32[2048, 768]" = torch.ops.aten.mm.default(view_43, permute_35);  permute_35 = None
        permute_36: "f32[768, 2048]" = torch.ops.aten.permute.default(view_43, [1, 0])
        mm_9: "f32[768, 768]" = torch.ops.aten.mm.default(permute_36, view);  permute_36 = None
        sum_15: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_43, [0], True);  view_43 = None
        view_44: "f32[768]" = torch.ops.aten.reshape.default(sum_15, [768]);  sum_15 = None
        view_45: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_8, [4, 512, 768]);  mm_8 = None
        add_12: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_11, view_45);  add_11 = view_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:207 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_39: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(mul_48, [0, 2, 1, 3]);  mul_48 = None
        clone_11: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_39, memory_format = torch.contiguous_format);  permute_39 = None
        view_46: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_11, [4, 512, 768]);  clone_11 = None
        view_47: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_46, [2048, 768]);  view_46 = None
        permute: "f32[768, 768]" = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
        permute_40: "f32[768, 768]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_10: "f32[2048, 768]" = torch.ops.aten.mm.default(view_47, permute_40);  permute_40 = None
        permute_41: "f32[768, 2048]" = torch.ops.aten.permute.default(view_47, [1, 0])
        mm_11: "f32[768, 768]" = torch.ops.aten.mm.default(permute_41, view);  permute_41 = view = None
        sum_16: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_47, [0], True);  view_47 = None
        view_48: "f32[768]" = torch.ops.aten.reshape.default(sum_16, [768]);  sum_16 = None
        view_49: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_10, [4, 512, 768]);  mm_10 = None
        add_13: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_12, view_49);  add_12 = view_49 = None
        return (add_13, mm_11, view_48, mm_9, view_44, mm_7, view_41, None, mm_5, view_29, sum_10, sum_11, mm_3, view_26, mm_1, view_23, sum_4, sum_5)

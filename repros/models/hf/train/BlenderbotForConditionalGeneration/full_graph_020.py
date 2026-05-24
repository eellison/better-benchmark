import torch
import torch._inductor.inductor_prims  # registers inductor prims ops
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[2560]", primals_3: "f32[16, 128, 2560]", primals_4: "f32[2560, 2560]", primals_6: "f32[2560, 2560]", primals_8: "f32[2560, 2560]", primals_11: "f32[2560, 2560]", primals_14: "f32[2560]", primals_16: "f32[2560, 2560]", primals_18: "f32[2560, 2560]", primals_20: "f32[2560, 2560]", primals_23: "f32[2560, 2560]", primals_25: "f32[2560]", primals_27: "f32[10240, 2560]", primals_29: "f32[2560, 10240]", getitem_1: "f32[16, 128, 1]", rsqrt: "f32[16, 128, 1]", view: "f32[2048, 2560]", permute_1: "f32[16, 32, 128, 80]", permute_4: "f32[16, 32, 128, 80]", permute_5: "f32[16, 32, 128, 80]", where: "f32[16, 1, 128, 128]", getitem_2: "f32[16, 32, 128, 80]", getitem_3: "f32[16, 32, 128]", getitem_4: "i64[]", getitem_5: "i64[]", addmm_3: "f32[2048, 2560]", gt: "b8[16, 128, 2560]", getitem_7: "f32[16, 128, 1]", rsqrt_1: "f32[16, 128, 1]", view_12: "f32[2048, 2560]", view_15: "f32[2048, 2560]", where_2: "f32[16, 32, 128, 128]", view_28: "f32[2048, 2560]", gt_1: "b8[16, 128, 2560]", mul_10: "f32[16, 128, 2560]", view_30: "f32[2048, 2560]", addmm_8: "f32[2048, 10240]", view_32: "f32[2048, 10240]", gt_2: "b8[16, 128, 2560]", div_1: "f32[16, 128, 1]", permute_33: "f32[512, 80, 128]", permute_34: "f32[512, 80, 128]", permute_35: "f32[512, 128, 80]", tangents_1: "f32[16, 128, 2560]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:391 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        convert_element_type: "f32[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(gt_2, torch.float32);  gt_2 = None
        mul_17: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.1111111111111112);  convert_element_type = None
        mul_18: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(tangents_1, mul_17);  mul_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_34: "f32[2048, 2560]" = torch.ops.aten.reshape.default(mul_18, [2048, 2560]);  mul_18 = None
        permute_18: "f32[10240, 2560]" = torch.ops.aten.permute.default(primals_29, [1, 0]);  primals_29 = None
        permute_19: "f32[2560, 10240]" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
        mm: "f32[2048, 10240]" = torch.ops.aten.mm.default(view_34, permute_19);  permute_19 = None
        permute_20: "f32[2560, 2048]" = torch.ops.aten.permute.default(view_34, [1, 0])
        mm_1: "f32[2560, 10240]" = torch.ops.aten.mm.default(permute_20, view_32);  permute_20 = view_32 = None
        sum_2: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_34, [0], True);  view_34 = None
        view_35: "f32[2560]" = torch.ops.aten.reshape.default(sum_2, [2560]);  sum_2 = None
        view_36: "f32[16, 128, 10240]" = torch.ops.aten.reshape.default(mm, [16, 128, 10240]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_31: "f32[16, 128, 10240]" = torch.ops.aten.reshape.default(addmm_8, [16, 128, 10240]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_13: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_31, 0.7071067811865476)
        erf: "f32[16, 128, 10240]" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
        add_9: "f32[16, 128, 10240]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_20: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(add_9, 0.5);  add_9 = None
        mul_21: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_31, view_31)
        mul_22: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_21, -0.5);  mul_21 = None
        exp_1: "f32[16, 128, 10240]" = torch.ops.aten.exp.default(mul_22);  mul_22 = None
        mul_23: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
        mul_24: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_31, mul_23);  view_31 = mul_23 = None
        add_12: "f32[16, 128, 10240]" = torch.ops.aten.add.Tensor(mul_20, mul_24);  mul_20 = mul_24 = None
        mul_25: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_36, add_12);  view_36 = add_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_37: "f32[2048, 10240]" = torch.ops.aten.reshape.default(mul_25, [2048, 10240]);  mul_25 = None
        permute_17: "f32[2560, 10240]" = torch.ops.aten.permute.default(primals_27, [1, 0]);  primals_27 = None
        permute_23: "f32[10240, 2560]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
        mm_2: "f32[2048, 2560]" = torch.ops.aten.mm.default(view_37, permute_23);  permute_23 = None
        permute_24: "f32[10240, 2048]" = torch.ops.aten.permute.default(view_37, [1, 0])
        mm_3: "f32[10240, 2560]" = torch.ops.aten.mm.default(permute_24, view_30);  permute_24 = view_30 = None
        sum_3: "f32[1, 10240]" = torch.ops.aten.sum.dim_IntList(view_37, [0], True);  view_37 = None
        view_38: "f32[10240]" = torch.ops.aten.reshape.default(sum_3, [10240]);  sum_3 = None
        view_39: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(mm_2, [16, 128, 2560]);  mm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        mul_27: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(view_39, primals_25);  primals_25 = None
        mul_28: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_27, 2560)
        sum_4: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_27, [2], True)
        mul_29: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_27, mul_10);  mul_27 = None
        sum_5: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_29, [2], True);  mul_29 = None
        mul_30: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_10, sum_5);  sum_5 = None
        sub_5: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(mul_28, sum_4);  mul_28 = sum_4 = None
        sub_6: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(sub_5, mul_30);  sub_5 = mul_30 = None
        mul_31: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(div_1, sub_6);  div_1 = sub_6 = None
        mul_32: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(view_39, mul_10);  mul_10 = None
        sum_6: "f32[2560]" = torch.ops.aten.sum.dim_IntList(mul_32, [0, 1]);  mul_32 = None
        sum_7: "f32[2560]" = torch.ops.aten.sum.dim_IntList(view_39, [0, 1]);  view_39 = None
        add_13: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(tangents_1, mul_31);  tangents_1 = mul_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:382 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        convert_element_type_1: "f32[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_33: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_34: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(add_13, mul_33);  mul_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_40: "f32[2048, 2560]" = torch.ops.aten.reshape.default(mul_34, [2048, 2560]);  mul_34 = None
        permute_16: "f32[2560, 2560]" = torch.ops.aten.permute.default(primals_23, [1, 0]);  primals_23 = None
        permute_27: "f32[2560, 2560]" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None
        mm_4: "f32[2048, 2560]" = torch.ops.aten.mm.default(view_40, permute_27);  permute_27 = None
        permute_28: "f32[2560, 2048]" = torch.ops.aten.permute.default(view_40, [1, 0])
        mm_5: "f32[2560, 2560]" = torch.ops.aten.mm.default(permute_28, view_28);  permute_28 = view_28 = None
        sum_8: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_40, [0], True);  view_40 = None
        view_41: "f32[2560]" = torch.ops.aten.reshape.default(sum_8, [2560]);  sum_8 = None
        view_42: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(mm_4, [16, 128, 2560]);  mm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_43: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_42, [16, 128, 32, 80]);  view_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_31: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_43, [0, 2, 1, 3]);  view_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_7: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(permute_31, memory_format = torch.contiguous_format);  permute_31 = None
        view_44: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_7, [512, 128, 80]);  clone_7 = None
        expand_3: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_2, [16, 32, 128, 128])
        view_24: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_3, [512, 128, 128]);  expand_3 = None
        permute_32: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_24, [0, 2, 1]);  view_24 = None
        bmm_2: "f32[512, 128, 80]" = torch.ops.aten.bmm.default(permute_32, view_44);  permute_32 = None
        bmm_3: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_44, permute_33);  view_44 = permute_33 = None
        view_45: "f32[16, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_2, [16, 32, 128, 80]);  bmm_2 = None
        view_46: "f32[16, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_3, [16, 32, 128, 128]);  bmm_3 = None
        mul_35: "f32[16, 32, 128, 128]" = torch.ops.aten.mul.Tensor(view_46, where_2);  view_46 = None
        sum_9: "f32[16, 32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_35, [-1], True)
        neg: "f32[16, 32, 128, 128]" = torch.ops.aten.neg.default(where_2);  where_2 = None
        fma: "f32[16, 32, 128, 128]" = torch.ops.prims.fma.default(neg, sum_9, mul_35);  neg = sum_9 = mul_35 = None
        view_47: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(fma, [512, 128, 128]);  fma = None
        bmm_4: "f32[512, 80, 128]" = torch.ops.aten.bmm.default(permute_34, view_47);  permute_34 = None
        bmm_5: "f32[512, 128, 80]" = torch.ops.aten.bmm.default(view_47, permute_35);  view_47 = permute_35 = None
        view_48: "f32[16, 32, 80, 128]" = torch.ops.aten.reshape.default(bmm_4, [16, 32, 80, 128]);  bmm_4 = None
        view_49: "f32[16, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_5, [16, 32, 128, 80]);  bmm_5 = None
        mul_36: "f32[16, 32, 80, 128]" = torch.ops.aten.mul.Scalar(view_48, 0.334370152488211);  view_48 = None
        permute_36: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(mul_36, [0, 1, 3, 2]);  mul_36 = None
        mul_37: "f32[16, 32, 128, 80]" = torch.ops.aten.mul.Scalar(view_49, 0.334370152488211);  view_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        permute_37: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(view_45, [0, 2, 1, 3]);  view_45 = None
        clone_8: "f32[16, 128, 32, 80]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
        view_50: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(clone_8, [16, 128, 2560]);  clone_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        permute_38: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(permute_36, [0, 2, 1, 3]);  permute_36 = None
        view_51: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(permute_38, [16, 128, 2560]);  permute_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_52: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_50, [2048, 2560]);  view_50 = None
        permute_11: "f32[2560, 2560]" = torch.ops.aten.permute.default(primals_20, [1, 0]);  primals_20 = None
        permute_39: "f32[2560, 2560]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        mm_6: "f32[2048, 2560]" = torch.ops.aten.mm.default(view_52, permute_39);  permute_39 = None
        permute_40: "f32[2560, 2048]" = torch.ops.aten.permute.default(view_52, [1, 0])
        mm_7: "f32[2560, 2560]" = torch.ops.aten.mm.default(permute_40, view_15);  permute_40 = None
        sum_10: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_52, [0], True);  view_52 = None
        view_53: "f32[2560]" = torch.ops.aten.reshape.default(sum_10, [2560]);  sum_10 = None
        view_54: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(mm_6, [16, 128, 2560]);  mm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        clone_9: "f32[16, 128, 2560]" = torch.ops.aten.clone.default(view_51, memory_format = torch.contiguous_format);  view_51 = None
        view_55: "f32[2048, 2560]" = torch.ops.aten.reshape.default(clone_9, [2048, 2560]);  clone_9 = None
        permute_10: "f32[2560, 2560]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
        permute_43: "f32[2560, 2560]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        mm_8: "f32[2048, 2560]" = torch.ops.aten.mm.default(view_55, permute_43);  permute_43 = None
        permute_44: "f32[2560, 2048]" = torch.ops.aten.permute.default(view_55, [1, 0])
        mm_9: "f32[2560, 2560]" = torch.ops.aten.mm.default(permute_44, view_15);  permute_44 = view_15 = None
        sum_11: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_55, [0], True);  view_55 = None
        view_56: "f32[2560]" = torch.ops.aten.reshape.default(sum_11, [2560]);  sum_11 = None
        view_57: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(mm_8, [16, 128, 2560]);  mm_8 = None
        add_14: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(view_54, view_57);  view_54 = view_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_47: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(mul_37, [0, 2, 1, 3]);  mul_37 = None
        clone_10: "f32[16, 128, 32, 80]" = torch.ops.aten.clone.default(permute_47, memory_format = torch.contiguous_format);  permute_47 = None
        view_58: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(clone_10, [16, 128, 2560]);  clone_10 = None
        view_59: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_58, [2048, 2560]);  view_58 = None
        permute_8: "f32[2560, 2560]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
        permute_48: "f32[2560, 2560]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        mm_10: "f32[2048, 2560]" = torch.ops.aten.mm.default(view_59, permute_48);  permute_48 = None
        permute_49: "f32[2560, 2048]" = torch.ops.aten.permute.default(view_59, [1, 0])
        mm_11: "f32[2560, 2560]" = torch.ops.aten.mm.default(permute_49, view_12);  permute_49 = view_12 = None
        sum_12: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_59, [0], True);  view_59 = None
        view_60: "f32[2560]" = torch.ops.aten.reshape.default(sum_12, [2560]);  sum_12 = None
        view_61: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(mm_10, [16, 128, 2560]);  mm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        mul_39: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(view_61, primals_14);  primals_14 = None
        mul_40: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_39, 2560)
        sum_13: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_39, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_11: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_3, [16, 128, 2560]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:367 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        mul_2: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(gt, view_11);  view_11 = None
        mul_3: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_2, 1.1111111111111112);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_2: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(primals_3, mul_3);  mul_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        sub_1: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_2, getitem_7);  add_2 = getitem_7 = None
        mul_4: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_41: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_39, mul_4);  mul_39 = None
        sum_14: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_41, [2], True);  mul_41 = None
        mul_42: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_4, sum_14);  sum_14 = None
        sub_8: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(mul_40, sum_13);  mul_40 = sum_13 = None
        sub_9: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(sub_8, mul_42);  sub_8 = mul_42 = None
        div_2: "f32[16, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 2560);  rsqrt_1 = None
        mul_43: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(div_2, sub_9);  div_2 = sub_9 = None
        mul_44: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(view_61, mul_4);  mul_4 = None
        sum_15: "f32[2560]" = torch.ops.aten.sum.dim_IntList(mul_44, [0, 1]);  mul_44 = None
        sum_16: "f32[2560]" = torch.ops.aten.sum.dim_IntList(view_61, [0, 1]);  view_61 = None
        add_15: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_13, mul_43);  add_13 = mul_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:367 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        convert_element_type_2: "f32[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_45: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
        mul_46: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(add_15, mul_45);  mul_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_62: "f32[2048, 2560]" = torch.ops.aten.reshape.default(mul_46, [2048, 2560]);  mul_46 = None
        permute_7: "f32[2560, 2560]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
        permute_52: "f32[2560, 2560]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
        mm_12: "f32[2048, 2560]" = torch.ops.aten.mm.default(view_62, permute_52);  permute_52 = None
        permute_53: "f32[2560, 2048]" = torch.ops.aten.permute.default(view_62, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_6: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_9: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(permute_6, [16, 128, -1]);  permute_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_10: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_9, [2048, 2560]);  view_9 = None
        mm_13: "f32[2560, 2560]" = torch.ops.aten.mm.default(permute_53, view_10);  permute_53 = view_10 = None
        sum_17: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_62, [0], True);  view_62 = None
        view_63: "f32[2560]" = torch.ops.aten.reshape.default(sum_17, [2560]);  sum_17 = None
        view_64: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(mm_12, [16, 128, 2560]);  mm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_65: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_64, [16, 128, 32, 80]);  view_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_56: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where, [16, 32, 128, 128]);  where = None
        _scaled_dot_product_efficient_attention_backward = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_56, permute_1, permute_4, permute_5, expand, getitem_2, getitem_3, getitem_4, getitem_5, 0.0, [True, True, True, False], scale = 0.11180339887498948);  permute_56 = permute_1 = permute_4 = permute_5 = expand = getitem_2 = getitem_3 = getitem_4 = getitem_5 = None
        getitem_10: "f32[16, 32, 128, 80]" = _scaled_dot_product_efficient_attention_backward[0]
        getitem_11: "f32[16, 32, 128, 80]" = _scaled_dot_product_efficient_attention_backward[1]
        getitem_12: "f32[16, 32, 128, 80]" = _scaled_dot_product_efficient_attention_backward[2];  _scaled_dot_product_efficient_attention_backward = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        permute_57: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(getitem_12, [0, 2, 1, 3]);  getitem_12 = None
        view_66: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(permute_57, [16, 128, 2560]);  permute_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        permute_58: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(getitem_11, [0, 2, 1, 3]);  getitem_11 = None
        view_67: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(permute_58, [16, 128, 2560]);  permute_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_68: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_66, [2048, 2560]);  view_66 = None
        permute_3: "f32[2560, 2560]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        permute_59: "f32[2560, 2560]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        mm_14: "f32[2048, 2560]" = torch.ops.aten.mm.default(view_68, permute_59);  permute_59 = None
        permute_60: "f32[2560, 2048]" = torch.ops.aten.permute.default(view_68, [1, 0])
        mm_15: "f32[2560, 2560]" = torch.ops.aten.mm.default(permute_60, view);  permute_60 = None
        sum_18: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_68, [0], True);  view_68 = None
        view_69: "f32[2560]" = torch.ops.aten.reshape.default(sum_18, [2560]);  sum_18 = None
        view_70: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(mm_14, [16, 128, 2560]);  mm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_71: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_67, [2048, 2560]);  view_67 = None
        permute_2: "f32[2560, 2560]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        permute_63: "f32[2560, 2560]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_16: "f32[2048, 2560]" = torch.ops.aten.mm.default(view_71, permute_63);  permute_63 = None
        permute_64: "f32[2560, 2048]" = torch.ops.aten.permute.default(view_71, [1, 0])
        mm_17: "f32[2560, 2560]" = torch.ops.aten.mm.default(permute_64, view);  permute_64 = None
        sum_19: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_71, [0], True);  view_71 = None
        view_72: "f32[2560]" = torch.ops.aten.reshape.default(sum_19, [2560]);  sum_19 = None
        view_73: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(mm_16, [16, 128, 2560]);  mm_16 = None
        add_16: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(view_70, view_73);  view_70 = view_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_67: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(getitem_10, [0, 2, 1, 3]);  getitem_10 = None
        view_74: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(permute_67, [16, 128, 2560]);  permute_67 = None
        view_75: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_74, [2048, 2560]);  view_74 = None
        permute: "f32[2560, 2560]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        permute_68: "f32[2560, 2560]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_18: "f32[2048, 2560]" = torch.ops.aten.mm.default(view_75, permute_68);  permute_68 = None
        permute_69: "f32[2560, 2048]" = torch.ops.aten.permute.default(view_75, [1, 0])
        mm_19: "f32[2560, 2560]" = torch.ops.aten.mm.default(permute_69, view);  permute_69 = view = None
        sum_20: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_75, [0], True);  view_75 = None
        view_76: "f32[2560]" = torch.ops.aten.reshape.default(sum_20, [2560]);  sum_20 = None
        view_77: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(mm_18, [16, 128, 2560]);  mm_18 = None
        add_17: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_16, view_77);  add_16 = view_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        mul_48: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(add_17, primals_1);  primals_1 = None
        mul_49: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_48, 2560)
        sum_21: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_48, [2], True)
        sub: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(primals_3, getitem_1);  primals_3 = getitem_1 = None
        mul: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_50: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_48, mul);  mul_48 = None
        sum_22: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_50, [2], True);  mul_50 = None
        mul_51: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul, sum_22);  sum_22 = None
        sub_11: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(mul_49, sum_21);  mul_49 = sum_21 = None
        sub_12: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(sub_11, mul_51);  sub_11 = mul_51 = None
        div_3: "f32[16, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt, 2560);  rsqrt = None
        mul_52: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(div_3, sub_12);  div_3 = sub_12 = None
        mul_53: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(add_17, mul);  mul = None
        sum_23: "f32[2560]" = torch.ops.aten.sum.dim_IntList(mul_53, [0, 1]);  mul_53 = None
        sum_24: "f32[2560]" = torch.ops.aten.sum.dim_IntList(add_17, [0, 1]);  add_17 = None
        add_18: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_15, mul_52);  add_15 = mul_52 = None
        return (sum_23, sum_24, add_18, mm_19, view_76, mm_17, view_72, mm_15, view_69, None, mm_13, view_63, add_14, sum_15, sum_16, mm_11, view_60, mm_9, view_56, mm_7, view_53, None, mm_5, view_41, sum_6, sum_7, mm_3, view_38, mm_1, view_35)

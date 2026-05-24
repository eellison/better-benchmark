import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[384]", primals_3: "f32[8, 1500, 384]", primals_4: "f32[384, 384]", primals_6: "f32[384, 384]", primals_7: "f32[384, 384]", primals_9: "f32[384, 384]", primals_11: "f32[384]", primals_13: "f32[1536, 384]", primals_15: "f32[384, 1536]", getitem_1: "f32[8, 1500, 1]", rsqrt: "f32[8, 1500, 1]", view: "f32[12000, 384]", clone_1: "f32[8, 6, 1500, 64]", clone_2: "f32[8, 6, 1500, 64]", clone_3: "f32[8, 6, 1500, 64]", getitem_2: "f32[8, 6, 1500, 64]", getitem_3: "f32[8, 6, 1504]", getitem_4: "i64[]", getitem_5: "i64[]", addmm_2: "f32[12000, 384]", getitem_7: "f32[8, 1500, 1]", rsqrt_1: "f32[8, 1500, 1]", view_12: "f32[12000, 384]", addmm_3: "f32[12000, 1536]", view_14: "f32[12000, 1536]", tangents_1: "f32[8, 1500, 384]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:405 in forward, code: hidden_states = self.fc2(hidden_states)
        clone_8: "f32[8, 1500, 384]" = torch.ops.aten.clone.default(tangents_1, memory_format = torch.contiguous_format)
        view_16: "f32[12000, 384]" = torch.ops.aten.reshape.default(clone_8, [12000, 384]);  clone_8 = None
        permute_9: "f32[1536, 384]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
        permute_10: "f32[384, 1536]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        mm_1: "f32[12000, 1536]" = torch.ops.aten.mm.default(view_16, permute_10);  permute_10 = None
        permute_11: "f32[384, 12000]" = torch.ops.aten.permute.default(view_16, [1, 0])
        mm_2: "f32[384, 1536]" = torch.ops.aten.mm.default(permute_11, view_14);  permute_11 = view_14 = None
        sum_1: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_16, [0], True);  view_16 = None
        view_17: "f32[384]" = torch.ops.aten.reshape.default(sum_1, [384]);  sum_1 = None
        view_18: "f32[8, 1500, 1536]" = torch.ops.aten.reshape.default(mm_1, [8, 1500, 1536]);  mm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:403 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_13: "f32[8, 1500, 1536]" = torch.ops.aten.reshape.default(addmm_3, [8, 1500, 1536]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_6: "f32[8, 1500, 1536]" = torch.ops.aten.mul.Tensor(view_13, 0.7071067811865476)
        erf: "f32[8, 1500, 1536]" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
        add_5: "f32[8, 1500, 1536]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_9: "f32[8, 1500, 1536]" = torch.ops.aten.mul.Tensor(add_5, 0.5);  add_5 = None
        mul_10: "f32[8, 1500, 1536]" = torch.ops.aten.mul.Tensor(view_13, view_13)
        mul_11: "f32[8, 1500, 1536]" = torch.ops.aten.mul.Tensor(mul_10, -0.5);  mul_10 = None
        exp: "f32[8, 1500, 1536]" = torch.ops.aten.exp.default(mul_11);  mul_11 = None
        mul_12: "f32[8, 1500, 1536]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_13: "f32[8, 1500, 1536]" = torch.ops.aten.mul.Tensor(view_13, mul_12);  view_13 = mul_12 = None
        add_8: "f32[8, 1500, 1536]" = torch.ops.aten.add.Tensor(mul_9, mul_13);  mul_9 = mul_13 = None
        mul_14: "f32[8, 1500, 1536]" = torch.ops.aten.mul.Tensor(view_18, add_8);  view_18 = add_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:403 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_19: "f32[12000, 1536]" = torch.ops.aten.reshape.default(mul_14, [12000, 1536]);  mul_14 = None
        permute_8: "f32[384, 1536]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
        permute_14: "f32[1536, 384]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        mm_3: "f32[12000, 384]" = torch.ops.aten.mm.default(view_19, permute_14);  permute_14 = None
        permute_15: "f32[1536, 12000]" = torch.ops.aten.permute.default(view_19, [1, 0])
        mm_4: "f32[1536, 384]" = torch.ops.aten.mm.default(permute_15, view_12);  permute_15 = view_12 = None
        sum_2: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_19, [0], True);  view_19 = None
        view_20: "f32[1536]" = torch.ops.aten.reshape.default(sum_2, [1536]);  sum_2 = None
        view_21: "f32[8, 1500, 384]" = torch.ops.aten.reshape.default(mm_3, [8, 1500, 384]);  mm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        view_11: "f32[8, 1500, 384]" = torch.ops.aten.reshape.default(addmm_2, [8, 1500, 384]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:399 in forward, code: hidden_states = residual + hidden_states
        add_2: "f32[8, 1500, 384]" = torch.ops.aten.add.Tensor(primals_3, view_11);  view_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:402 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        sub_2: "f32[8, 1500, 384]" = torch.ops.aten.sub.Tensor(add_2, getitem_7);  add_2 = getitem_7 = None
        mul_15: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
        mul_16: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(view_21, primals_11);  primals_11 = None
        mul_17: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(mul_16, 384)
        sum_3: "f32[8, 1500, 1]" = torch.ops.aten.sum.dim_IntList(mul_16, [2], True)
        mul_18: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(mul_16, mul_15);  mul_16 = None
        sum_4: "f32[8, 1500, 1]" = torch.ops.aten.sum.dim_IntList(mul_18, [2], True);  mul_18 = None
        mul_19: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(mul_15, sum_4);  sum_4 = None
        sub_3: "f32[8, 1500, 384]" = torch.ops.aten.sub.Tensor(mul_17, sum_3);  mul_17 = sum_3 = None
        sub_4: "f32[8, 1500, 384]" = torch.ops.aten.sub.Tensor(sub_3, mul_19);  sub_3 = mul_19 = None
        div: "f32[8, 1500, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 384);  rsqrt_1 = None
        mul_20: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(div, sub_4);  div = sub_4 = None
        mul_21: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(view_21, mul_15);  mul_15 = None
        sum_5: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_21, [0, 1]);  mul_21 = None
        sum_6: "f32[384]" = torch.ops.aten.sum.dim_IntList(view_21, [0, 1]);  view_21 = None
        add_9: "f32[8, 1500, 384]" = torch.ops.aten.add.Tensor(tangents_1, mul_20);  tangents_1 = mul_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        clone_9: "f32[8, 1500, 384]" = torch.ops.aten.clone.default(add_9, memory_format = torch.contiguous_format)
        view_22: "f32[12000, 384]" = torch.ops.aten.reshape.default(clone_9, [12000, 384]);  clone_9 = None
        permute_7: "f32[384, 384]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
        permute_18: "f32[384, 384]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
        mm_5: "f32[12000, 384]" = torch.ops.aten.mm.default(view_22, permute_18);  permute_18 = None
        permute_19: "f32[384, 12000]" = torch.ops.aten.permute.default(view_22, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_6: "f32[8, 1500, 6, 64]" = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:353 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_9: "f32[8, 1500, 384]" = torch.ops.aten.reshape.default(permute_6, [8, 1500, -1]);  permute_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        view_10: "f32[12000, 384]" = torch.ops.aten.reshape.default(view_9, [12000, 384]);  view_9 = None
        mm_6: "f32[384, 384]" = torch.ops.aten.mm.default(permute_19, view_10);  permute_19 = view_10 = None
        sum_7: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_22, [0], True);  view_22 = None
        view_23: "f32[384]" = torch.ops.aten.reshape.default(sum_7, [384]);  sum_7 = None
        view_24: "f32[8, 1500, 384]" = torch.ops.aten.reshape.default(mm_5, [8, 1500, 384]);  mm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:353 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_25: "f32[8, 1500, 6, 64]" = torch.ops.aten.reshape.default(view_24, [8, 1500, 6, 64]);  view_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_22: "f32[8, 6, 1500, 64]" = torch.ops.aten.permute.default(view_25, [0, 2, 1, 3]);  view_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_22, clone_1, clone_2, clone_3, None, getitem_2, getitem_3, getitem_4, getitem_5, 0.0, [True, True, True, False], scale = 1.0);  permute_22 = clone_1 = clone_2 = clone_3 = getitem_2 = getitem_3 = getitem_4 = getitem_5 = None
        getitem_8: "f32[8, 6, 1500, 64]" = _scaled_dot_product_efficient_attention_backward[0]
        getitem_9: "f32[8, 6, 1500, 64]" = _scaled_dot_product_efficient_attention_backward[1]
        getitem_10: "f32[8, 6, 1500, 64]" = _scaled_dot_product_efficient_attention_backward[2];  _scaled_dot_product_efficient_attention_backward = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        permute_23: "f32[8, 1500, 6, 64]" = torch.ops.aten.permute.default(getitem_10, [0, 2, 1, 3]);  getitem_10 = None
        view_26: "f32[8, 1500, 384]" = torch.ops.aten.reshape.default(permute_23, [8, 1500, 384]);  permute_23 = None
        view_27: "f32[12000, 384]" = torch.ops.aten.reshape.default(view_26, [12000, 384]);  view_26 = None
        permute_4: "f32[384, 384]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
        permute_24: "f32[384, 384]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        mm_7: "f32[12000, 384]" = torch.ops.aten.mm.default(view_27, permute_24);  permute_24 = None
        permute_25: "f32[384, 12000]" = torch.ops.aten.permute.default(view_27, [1, 0])
        mm_8: "f32[384, 384]" = torch.ops.aten.mm.default(permute_25, view);  permute_25 = None
        sum_8: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_27, [0], True);  view_27 = None
        view_28: "f32[384]" = torch.ops.aten.reshape.default(sum_8, [384]);  sum_8 = None
        view_29: "f32[8, 1500, 384]" = torch.ops.aten.reshape.default(mm_7, [8, 1500, 384]);  mm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        permute_28: "f32[8, 1500, 6, 64]" = torch.ops.aten.permute.default(getitem_9, [0, 2, 1, 3]);  getitem_9 = None
        view_30: "f32[8, 1500, 384]" = torch.ops.aten.reshape.default(permute_28, [8, 1500, 384]);  permute_28 = None
        view_31: "f32[12000, 384]" = torch.ops.aten.reshape.default(view_30, [12000, 384]);  view_30 = None
        permute_29: "f32[384, 12000]" = torch.ops.aten.permute.default(view_31, [1, 0])
        mm_9: "f32[384, 384]" = torch.ops.aten.mm.default(permute_29, view);  permute_29 = None
        permute_2: "f32[384, 384]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        permute_31: "f32[384, 384]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_10: "f32[12000, 384]" = torch.ops.aten.mm.default(view_31, permute_31);  view_31 = permute_31 = None
        view_32: "f32[8, 1500, 384]" = torch.ops.aten.reshape.default(mm_10, [8, 1500, 384]);  mm_10 = None
        add_10: "f32[8, 1500, 384]" = torch.ops.aten.add.Tensor(view_29, view_32);  view_29 = view_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        permute_33: "f32[8, 1500, 6, 64]" = torch.ops.aten.permute.default(getitem_8, [0, 2, 1, 3]);  getitem_8 = None
        view_33: "f32[8, 1500, 384]" = torch.ops.aten.reshape.default(permute_33, [8, 1500, 384]);  permute_33 = None
        mul_22: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(view_33, 0.125);  view_33 = None
        view_34: "f32[12000, 384]" = torch.ops.aten.reshape.default(mul_22, [12000, 384]);  mul_22 = None
        permute: "f32[384, 384]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        permute_34: "f32[384, 384]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_11: "f32[12000, 384]" = torch.ops.aten.mm.default(view_34, permute_34);  permute_34 = None
        permute_35: "f32[384, 12000]" = torch.ops.aten.permute.default(view_34, [1, 0])
        mm_12: "f32[384, 384]" = torch.ops.aten.mm.default(permute_35, view);  permute_35 = view = None
        sum_9: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_34, [0], True);  view_34 = None
        view_35: "f32[384]" = torch.ops.aten.reshape.default(sum_9, [384]);  sum_9 = None
        view_36: "f32[8, 1500, 384]" = torch.ops.aten.reshape.default(mm_11, [8, 1500, 384]);  mm_11 = None
        add_11: "f32[8, 1500, 384]" = torch.ops.aten.add.Tensor(add_10, view_36);  add_10 = view_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:392 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        sub_5: "f32[8, 1500, 384]" = torch.ops.aten.sub.Tensor(primals_3, getitem_1);  primals_3 = getitem_1 = None
        mul_23: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt);  sub_5 = None
        mul_24: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(add_11, primals_1);  primals_1 = None
        mul_25: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(mul_24, 384)
        sum_10: "f32[8, 1500, 1]" = torch.ops.aten.sum.dim_IntList(mul_24, [2], True)
        mul_26: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(mul_24, mul_23);  mul_24 = None
        sum_11: "f32[8, 1500, 1]" = torch.ops.aten.sum.dim_IntList(mul_26, [2], True);  mul_26 = None
        mul_27: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(mul_23, sum_11);  sum_11 = None
        sub_6: "f32[8, 1500, 384]" = torch.ops.aten.sub.Tensor(mul_25, sum_10);  mul_25 = sum_10 = None
        sub_7: "f32[8, 1500, 384]" = torch.ops.aten.sub.Tensor(sub_6, mul_27);  sub_6 = mul_27 = None
        div_1: "f32[8, 1500, 1]" = torch.ops.aten.div.Tensor(rsqrt, 384);  rsqrt = None
        mul_28: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(div_1, sub_7);  div_1 = sub_7 = None
        mul_29: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(add_11, mul_23);  mul_23 = None
        sum_12: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_29, [0, 1]);  mul_29 = None
        sum_13: "f32[384]" = torch.ops.aten.sum.dim_IntList(add_11, [0, 1]);  add_11 = None
        add_12: "f32[8, 1500, 384]" = torch.ops.aten.add.Tensor(add_9, mul_28);  add_9 = mul_28 = None
        return (sum_12, sum_13, add_12, mm_12, view_35, mm_9, mm_8, view_28, mm_6, view_23, sum_5, sum_6, mm_4, view_20, mm_2, view_17)

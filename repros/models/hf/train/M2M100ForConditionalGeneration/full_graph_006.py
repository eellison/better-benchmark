import torch
import torch._inductor.inductor_prims  # registers inductor prims ops
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[1024]", primals_2: "f32[1024]", primals_3: "f32[64, 128, 1024]", primals_4: "f32[1024, 1024]", primals_5: "f32[1024]", primals_6: "f32[1024, 1024]", primals_7: "f32[1024]", primals_8: "f32[1024, 1024]", primals_9: "f32[1024]", primals_10: "b8[64, 1, 128, 128]", primals_11: "f32[1024, 1024]", primals_12: "f32[1024]", primals_13: "f32[64, 128, 1024]", primals_14: "f32[1024]", primals_15: "f32[1024]", primals_16: "f32[1024, 1024]", primals_17: "f32[1024]", primals_18: "f32[1024, 1024]", primals_19: "f32[1024]", primals_20: "f32[1024, 1024]", primals_21: "f32[1024]", primals_22: "b8[64, 1, 128, 128]", primals_23: "f32[1024, 1024]", primals_24: "f32[1024]", primals_25: "f32[1024]", primals_26: "f32[1024]", primals_27: "f32[4096, 1024]", primals_28: "f32[4096]", primals_29: "f32[1024, 4096]", primals_30: "f32[1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:441 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean = torch.ops.aten.var_mean.correction(primals_3, [2], correction = 0, keepdim = True)
        getitem: "f32[64, 128, 1]" = var_mean[0]
        getitem_1: "f32[64, 128, 1]" = var_mean[1];  var_mean = None
        add: "f32[64, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[64, 128, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(primals_3, getitem_1)
        mul: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul, primals_1);  mul = None
        add_1: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(mul_1, primals_2);  mul_1 = primals_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_1, [8192, 1024]);  add_1 = None
        permute: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_4, [1, 0])
        addmm: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_5, view, permute);  primals_5 = permute = None
        view_1: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(addmm, [64, 128, 1024]);  addmm = None
        view_2: "f32[64, 128, 16, 64]" = torch.ops.aten.reshape.default(view_1, [64, 128, -1, 64]);  view_1 = None
        permute_1: "f32[64, 16, 128, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        permute_2: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_6, [1, 0])
        addmm_1: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_7, view, permute_2);  primals_7 = permute_2 = None
        view_4: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(addmm_1, [64, 128, 1024]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        permute_3: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_8, [1, 0])
        addmm_2: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_9, view, permute_3);  primals_9 = permute_3 = None
        view_6: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(addmm_2, [64, 128, 1024]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:299 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_7: "f32[64, 128, 16, 64]" = torch.ops.aten.reshape.default(view_4, [64, 128, -1, 64]);  view_4 = None
        permute_4: "f32[64, 16, 128, 64]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_8: "f32[64, 128, 16, 64]" = torch.ops.aten.reshape.default(view_6, [64, 128, -1, 64]);  view_6 = None
        permute_5: "f32[64, 16, 128, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[64, 1, 128, 128]" = torch.ops.aten.where.self(primals_10, full_default_1, full_default);  primals_10 = None
        expand: "f32[64, 16, 128, 128]" = torch.ops.aten.expand.default(where, [64, 16, 128, 128])
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_1, permute_4, permute_5, expand, True, 0.1, scale = 0.125);  expand = None
        getitem_2: "f32[64, 16, 128, 64]" = _scaled_dot_product_efficient_attention[0]
        getitem_3: "f32[64, 16, 128]" = _scaled_dot_product_efficient_attention[1]
        getitem_4: "i64[]" = _scaled_dot_product_efficient_attention[2]
        getitem_5: "i64[]" = _scaled_dot_product_efficient_attention[3];  _scaled_dot_product_efficient_attention = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_6: "f32[64, 128, 16, 64]" = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_9: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(permute_6, [64, 128, -1]);  permute_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_10: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_9, [8192, 1024]);  view_9 = None
        permute_7: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_11, [1, 0])
        addmm_3: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_12, view_10, permute_7);  primals_12 = view_10 = permute_7 = None
        view_11: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(addmm_3, [64, 128, 1024])

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[4]" = torch.ops.prims.inductor_seeds.default(4, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:450 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_3: "f32[64, 128, 1024]" = torch.ops.prims.inductor_random.default([64, 128, 1024], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt: "b8[64, 128, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_3, 0.1);  inductor_random_default_3 = None
        mul_2: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(gt, view_11);  view_11 = None
        mul_3: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_2, 1.1111111111111112);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:451 in forward, code: hidden_states = residual + hidden_states
        add_2: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(primals_3, mul_3);  mul_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:456 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        var_mean_1 = torch.ops.aten.var_mean.correction(add_2, [2], correction = 0, keepdim = True)
        getitem_6: "f32[64, 128, 1]" = var_mean_1[0]
        getitem_7: "f32[64, 128, 1]" = var_mean_1[1];  var_mean_1 = None
        add_3: "f32[64, 128, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_1: "f32[64, 128, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        sub_1: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(add_2, getitem_7)
        mul_4: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_5: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_4, primals_14);  mul_4 = None
        add_4: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(mul_5, primals_15);  mul_5 = primals_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_12: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_4, [8192, 1024]);  add_4 = None
        permute_8: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_16, [1, 0])
        addmm_4: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_17, view_12, permute_8);  primals_17 = permute_8 = None
        view_13: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(addmm_4, [64, 128, 1024]);  addmm_4 = None
        view_14: "f32[64, 128, 16, 64]" = torch.ops.aten.reshape.default(view_13, [64, 128, -1, 64]);  view_13 = None
        permute_9: "f32[64, 16, 128, 64]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        view_15: "f32[8192, 1024]" = torch.ops.aten.reshape.default(primals_13, [8192, 1024]);  primals_13 = None
        permute_10: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_18, [1, 0])
        addmm_5: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_19, view_15, permute_10);  primals_19 = permute_10 = None
        view_16: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(addmm_5, [64, 128, 1024]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:297 in forward, code: value_states = self.v_proj(current_states)
        permute_11: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_20, [1, 0])
        addmm_6: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_21, view_15, permute_11);  primals_21 = permute_11 = None
        view_18: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(addmm_6, [64, 128, 1024]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:299 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_19: "f32[64, 128, 16, 64]" = torch.ops.aten.reshape.default(view_16, [64, 128, -1, 64]);  view_16 = None
        permute_12: "f32[64, 16, 128, 64]" = torch.ops.aten.permute.default(view_19, [0, 2, 1, 3]);  view_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:300 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_20: "f32[64, 128, 16, 64]" = torch.ops.aten.reshape.default(view_18, [64, 128, -1, 64]);  view_18 = None
        permute_13: "f32[64, 16, 128, 64]" = torch.ops.aten.permute.default(view_20, [0, 2, 1, 3]);  view_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        where_1: "f32[64, 1, 128, 128]" = torch.ops.aten.where.self(primals_22, full_default_1, full_default);  primals_22 = full_default_1 = full_default = None
        mul_6: "f32[64, 16, 128, 64]" = torch.ops.aten.mul.Scalar(permute_9, 0.3535533905932738);  permute_9 = None
        permute_14: "f32[64, 16, 64, 128]" = torch.ops.aten.permute.default(permute_12, [0, 1, 3, 2]);  permute_12 = None
        mul_7: "f32[64, 16, 64, 128]" = torch.ops.aten.mul.Scalar(permute_14, 0.3535533905932738);  permute_14 = None
        expand_1: "f32[64, 16, 128, 64]" = torch.ops.aten.expand.default(mul_6, [64, 16, 128, 64]);  mul_6 = None
        clone: "f32[64, 16, 128, 64]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_21: "f32[1024, 128, 64]" = torch.ops.aten.reshape.default(clone, [1024, 128, 64]);  clone = None
        expand_2: "f32[64, 16, 64, 128]" = torch.ops.aten.expand.default(mul_7, [64, 16, 64, 128]);  mul_7 = None
        clone_1: "f32[64, 16, 64, 128]" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        view_22: "f32[1024, 64, 128]" = torch.ops.aten.reshape.default(clone_1, [1024, 64, 128]);  clone_1 = None
        bmm: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_21, view_22)
        view_23: "f32[64, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm, [64, 16, 128, 128]);  bmm = None
        add_5: "f32[64, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_23, where_1);  view_23 = where_1 = None
        amax: "f32[64, 16, 128, 1]" = torch.ops.aten.amax.default(add_5, [-1], True)
        sub_2: "f32[64, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_5, amax);  amax = None
        exp: "f32[64, 16, 128, 128]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_1: "f32[64, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[64, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        eq: "b8[64, 16, 128, 128]" = torch.ops.aten.eq.Scalar(add_5, -inf);  add_5 = None
        logical_not: "b8[64, 16, 128, 128]" = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1: "b8[64, 16, 128, 1]" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1: "b8[64, 16, 128, 1]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        full_default_4: "f32[64, 16, 128, 128]" = torch.ops.aten.full.default([64, 16, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f32[64, 16, 128, 128]" = torch.ops.aten.where.self(logical_not_1, full_default_4, div);  logical_not_1 = full_default_4 = div = None
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default_2: "f32[64, 16, 128, 128]" = torch.ops.prims.inductor_random.default([64, 16, 128, 128], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        gt_1: "b8[64, 16, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_2, 0.1);  inductor_random_default_2 = None
        mul_8: "f32[64, 16, 128, 128]" = torch.ops.aten.mul.Tensor(gt_1, where_2)
        mul_9: "f32[64, 16, 128, 128]" = torch.ops.aten.mul.Tensor(mul_8, 1.1111111111111112);  mul_8 = None
        expand_3: "f32[64, 16, 128, 128]" = torch.ops.aten.expand.default(mul_9, [64, 16, 128, 128]);  mul_9 = None
        view_24: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_3, [1024, 128, 128]);  expand_3 = None
        expand_4: "f32[64, 16, 128, 64]" = torch.ops.aten.expand.default(permute_13, [64, 16, 128, 64]);  permute_13 = None
        clone_2: "f32[64, 16, 128, 64]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_25: "f32[1024, 128, 64]" = torch.ops.aten.reshape.default(clone_2, [1024, 128, 64]);  clone_2 = None
        bmm_1: "f32[1024, 128, 64]" = torch.ops.aten.bmm.default(view_24, view_25)
        view_26: "f32[64, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_1, [64, 16, 128, 64]);  bmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_15: "f32[64, 128, 16, 64]" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
        clone_3: "f32[64, 128, 16, 64]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_27: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(clone_3, [64, 128, -1]);  clone_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        view_28: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_27, [8192, 1024]);  view_27 = None
        permute_16: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_23, [1, 0])
        addmm_7: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_24, view_28, permute_16);  primals_24 = permute_16 = None
        view_29: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(addmm_7, [64, 128, 1024]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:465 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_2: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2)
        inductor_random_default_1: "f32[64, 128, 1024]" = torch.ops.prims.inductor_random.default([64, 128, 1024], inductor_lookup_seed_default_2, 'rand');  inductor_lookup_seed_default_2 = None
        gt_2: "b8[64, 128, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default_1, 0.1);  inductor_random_default_1 = None
        mul_10: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(gt_2, view_29);  view_29 = None
        mul_11: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_10, 1.1111111111111112);  mul_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:466 in forward, code: hidden_states = residual + hidden_states
        add_6: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(add_2, mul_11);  add_2 = mul_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:470 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_2 = torch.ops.aten.var_mean.correction(add_6, [2], correction = 0, keepdim = True)
        getitem_8: "f32[64, 128, 1]" = var_mean_2[0]
        getitem_9: "f32[64, 128, 1]" = var_mean_2[1];  var_mean_2 = None
        add_7: "f32[64, 128, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_2: "f32[64, 128, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        sub_3: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(add_6, getitem_9);  getitem_9 = None
        mul_12: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
        mul_13: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_12, primals_25)
        add_8: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(mul_13, primals_26);  mul_13 = primals_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:471 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_30: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_8, [8192, 1024]);  add_8 = None
        permute_17: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_27, [1, 0])
        addmm_8: "f32[8192, 4096]" = torch.ops.aten.addmm.default(primals_28, view_30, permute_17);  primals_28 = permute_17 = None
        view_31: "f32[64, 128, 4096]" = torch.ops.aten.reshape.default(addmm_8, [64, 128, 4096]);  addmm_8 = None
        relu: "f32[64, 128, 4096]" = torch.ops.aten.relu.default(view_31);  view_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:473 in forward, code: hidden_states = self.fc2(hidden_states)
        view_32: "f32[8192, 4096]" = torch.ops.aten.reshape.default(relu, [8192, 4096])
        permute_18: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_29, [1, 0])
        addmm_9: "f32[8192, 1024]" = torch.ops.aten.addmm.default(primals_30, view_32, permute_18);  primals_30 = permute_18 = None
        view_33: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(addmm_9, [64, 128, 1024]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:474 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_3: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 3);  inductor_seeds_default = None
        inductor_random_default: "f32[64, 128, 1024]" = torch.ops.prims.inductor_random.default([64, 128, 1024], inductor_lookup_seed_default_3, 'rand');  inductor_lookup_seed_default_3 = None
        gt_3: "b8[64, 128, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_14: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(gt_3, view_33);  view_33 = None
        mul_15: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_14, 1.1111111111111112);  mul_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:475 in forward, code: hidden_states = residual + hidden_states
        add_9: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(add_6, mul_15);  add_6 = mul_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:471 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        le: "b8[64, 128, 4096]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:470 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        div_1: "f32[64, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 1024);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_32: "f32[1024, 128, 128]" = torch.ops.aten.permute.default(view_24, [0, 2, 1]);  view_24 = None
        permute_33: "f32[1024, 64, 128]" = torch.ops.aten.permute.default(view_25, [0, 2, 1]);  view_25 = None
        permute_34: "f32[1024, 64, 128]" = torch.ops.aten.permute.default(view_21, [0, 2, 1]);  view_21 = None
        permute_35: "f32[1024, 128, 64]" = torch.ops.aten.permute.default(view_22, [0, 2, 1]);  view_22 = None
        return (add_9, primals_1, primals_3, primals_4, primals_6, primals_8, primals_11, primals_14, primals_16, primals_18, primals_20, primals_23, primals_25, primals_27, primals_29, getitem_1, rsqrt, view, permute_1, permute_4, permute_5, where, getitem_2, getitem_3, getitem_4, getitem_5, addmm_3, gt, getitem_7, rsqrt_1, view_12, view_15, where_2, gt_1, view_28, gt_2, mul_12, view_30, view_32, gt_3, le, div_1, permute_32, permute_33, permute_34, permute_35)

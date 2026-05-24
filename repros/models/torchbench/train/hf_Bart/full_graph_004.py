import torch
import torch._inductor.inductor_prims  # registers inductor prims ops
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[4, 512, 768]", primals_2: "f32[768, 768]", primals_3: "f32[768]", primals_4: "f32[768, 768]", primals_5: "f32[768]", primals_6: "f32[768, 768]", primals_7: "f32[768]", primals_8: "b8[4, 1, 512, 512]", primals_9: "f32[768, 768]", primals_10: "f32[768]", primals_11: "f32[768]", primals_12: "f32[768]", primals_13: "f32[3072, 768]", primals_14: "f32[3072]", primals_15: "f32[768, 3072]", primals_16: "f32[768]", primals_17: "f32[768]", primals_18: "f32[768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:207 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view: "f32[2048, 768]" = torch.ops.aten.reshape.default(primals_1, [2048, 768])
        permute: "f32[768, 768]" = torch.ops.aten.permute.default(primals_2, [1, 0])
        addmm: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_3, view, permute);  primals_3 = permute = None
        view_1: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm, [4, 512, 768]);  addmm = None
        view_2: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_1, [4, 512, -1, 64]);  view_1 = None
        permute_1: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:227 in forward, code: key_states = self.k_proj(current_states)
        permute_2: "f32[768, 768]" = torch.ops.aten.permute.default(primals_4, [1, 0])
        addmm_1: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_5, view, permute_2);  primals_5 = permute_2 = None
        view_4: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_1, [4, 512, 768]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:228 in forward, code: value_states = self.v_proj(current_states)
        permute_3: "f32[768, 768]" = torch.ops.aten.permute.default(primals_6, [1, 0])
        addmm_2: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_7, view, permute_3);  primals_7 = view = permute_3 = None
        view_6: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_2, [4, 512, 768]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:230 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_7: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_4, [4, 512, -1, 64]);  view_4 = None
        permute_4: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:231 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_8: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_6, [4, 512, -1, 64]);  view_6 = None
        permute_5: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[4, 1, 512, 512]" = torch.ops.aten.where.self(primals_8, full_default_1, full_default);  full_default_1 = full_default = None
        mul: "f32[4, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_1, 0.3535533905932738);  permute_1 = None
        permute_6: "f32[4, 12, 64, 512]" = torch.ops.aten.permute.default(permute_4, [0, 1, 3, 2]);  permute_4 = None
        mul_1: "f32[4, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_6, 0.3535533905932738);  permute_6 = None
        expand: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(mul, [4, 12, 512, 64]);  mul = None
        clone: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_9: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone, [48, 512, 64]);  clone = None
        expand_1: "f32[4, 12, 64, 512]" = torch.ops.aten.expand.default(mul_1, [4, 12, 64, 512]);  mul_1 = None
        clone_1: "f32[4, 12, 64, 512]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_10: "f32[48, 64, 512]" = torch.ops.aten.reshape.default(clone_1, [48, 64, 512]);  clone_1 = None
        bmm: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_9, view_10)
        view_11: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm, [4, 12, 512, 512])
        add: "f32[4, 12, 512, 512]" = torch.ops.aten.add.Tensor(view_11, where);  view_11 = where = None
        amax: "f32[4, 12, 512, 1]" = torch.ops.aten.amax.default(add, [-1], True)
        sub: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(add, amax)
        exp: "f32[4, 12, 512, 512]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None
        eq: "b8[4, 12, 512, 512]" = torch.ops.aten.eq.Scalar(add, -inf);  add = None
        logical_not: "b8[4, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1: "b8[4, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1: "b8[4, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        full_default_2: "f32[4, 12, 512, 512]" = torch.ops.aten.full.default([4, 12, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[4, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_1, full_default_2, div);  full_default_2 = div = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[4]" = torch.ops.prims.inductor_seeds.default(4, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_3: "f32[4, 12, 512, 512]" = torch.ops.prims.inductor_random.default([4, 12, 512, 512], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt: "b8[4, 12, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_3, 0.1);  inductor_random_default_3 = None
        mul_2: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(gt, where_1);  where_1 = None
        mul_3: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_2, 1.1111111111111112);  mul_2 = None
        expand_2: "f32[4, 12, 512, 512]" = torch.ops.aten.expand.default(mul_3, [4, 12, 512, 512]);  mul_3 = None
        view_12: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(expand_2, [48, 512, 512]);  expand_2 = None
        expand_3: "f32[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_5, [4, 12, 512, 64]);  permute_5 = None
        clone_2: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_13: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_2, [48, 512, 64]);  clone_2 = None
        bmm_1: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_12, view_13)
        view_14: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_1, [4, 12, 512, 64]);  bmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
        clone_3: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:254 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_15: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_3, [4, 512, -1]);  clone_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:255 in forward, code: attn_output = self.out_proj(attn_output)
        view_16: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_15, [2048, 768]);  view_15 = None
        permute_8: "f32[768, 768]" = torch.ops.aten.permute.default(primals_9, [1, 0])
        addmm_3: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_10, view_16, permute_8);  primals_10 = permute_8 = None
        view_17: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_3, [4, 512, 768])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:292 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default_2: "f32[4, 512, 768]" = torch.ops.prims.inductor_random.default([4, 512, 768], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        gt_1: "b8[4, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_2, 0.1);  inductor_random_default_2 = None
        mul_4: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(gt_1, view_17);  view_17 = None
        mul_5: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_4, 1.1111111111111112);  mul_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:293 in forward, code: hidden_states = residual + hidden_states
        add_1: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(primals_1, mul_5);  mul_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:294 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
        getitem: "f32[4, 512, 1]" = var_mean[0]
        getitem_1: "f32[4, 512, 1]" = var_mean[1];  var_mean = None
        add_2: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub_1: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = None
        mul_6: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = None
        mul_7: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_6, primals_11);  mul_6 = None
        add_3: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_7, primals_12);  mul_7 = primals_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:297 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_18: "f32[2048, 768]" = torch.ops.aten.reshape.default(add_3, [2048, 768])
        permute_9: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_13, [1, 0])
        addmm_4: "f32[2048, 3072]" = torch.ops.aten.addmm.default(primals_14, view_18, permute_9);  primals_14 = permute_9 = None
        view_19: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_4, [4, 512, 3072])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_8: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.5)
        mul_9: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476);  view_19 = None
        erf: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_9);  mul_9 = None
        add_4: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_10: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_8, add_4);  mul_8 = add_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:298 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        inductor_lookup_seed_default_2: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2)
        inductor_random_default_1: "f32[4, 512, 3072]" = torch.ops.prims.inductor_random.default([4, 512, 3072], inductor_lookup_seed_default_2, 'rand');  inductor_lookup_seed_default_2 = None
        gt_2: "b8[4, 512, 3072]" = torch.ops.aten.gt.Scalar(inductor_random_default_1, 0.1);  inductor_random_default_1 = None
        mul_11: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(gt_2, mul_10);  mul_10 = None
        mul_12: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_11, 1.1111111111111112);  mul_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:299 in forward, code: hidden_states = self.fc2(hidden_states)
        view_20: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_12, [2048, 3072]);  mul_12 = None
        permute_10: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_15, [1, 0])
        addmm_5: "f32[2048, 768]" = torch.ops.aten.addmm.default(primals_16, view_20, permute_10);  primals_16 = permute_10 = None
        view_21: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_5, [4, 512, 768]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:300 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_3: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 3);  inductor_seeds_default = None
        inductor_random_default: "f32[4, 512, 768]" = torch.ops.prims.inductor_random.default([4, 512, 768], inductor_lookup_seed_default_3, 'rand');  inductor_lookup_seed_default_3 = None
        gt_3: "b8[4, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_13: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(gt_3, view_21);  view_21 = None
        mul_14: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_13, 1.1111111111111112);  mul_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:301 in forward, code: hidden_states = residual + hidden_states
        add_5: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_3, mul_14);  add_3 = mul_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:302 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_1 = torch.ops.aten.var_mean.correction(add_5, [2], correction = 0, keepdim = True)
        getitem_2: "f32[4, 512, 1]" = var_mean_1[0]
        getitem_3: "f32[4, 512, 1]" = var_mean_1[1];  var_mean_1 = None
        add_6: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_2: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(add_5, getitem_3);  add_5 = getitem_3 = None
        mul_15: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
        mul_16: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_15, primals_17)
        add_7: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_16, primals_18);  mul_16 = primals_18 = None
        div_1: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_24: "f32[48, 512, 512]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
        permute_25: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None
        permute_26: "f32[48, 64, 512]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
        permute_27: "f32[48, 512, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
        return (add_7, primals_1, primals_2, primals_4, primals_6, primals_8, primals_9, primals_11, primals_13, primals_15, primals_17, bmm, amax, sum_1, logical_not_1, gt, view_16, addmm_3, gt_1, getitem_1, rsqrt, view_18, addmm_4, gt_2, view_20, gt_3, mul_15, div_1, permute_24, permute_25, permute_26, permute_27)

import torch
import torch._inductor.inductor_prims  # registers inductor prims ops
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f16[768]", arg1_1: "f16[768]", arg2_1: "f16[4, 512, 768]", arg3_1: "f16[768, 768]", arg4_1: "f16[768]", arg5_1: "f16[768, 768]", arg6_1: "f16[768]", arg7_1: "f16[768, 768]", arg8_1: "f16[768]", arg9_1: "b8[4, 1, 512, 512]", arg10_1: "f16[768, 768]", arg11_1: "f16[768]", arg12_1: "f16[768]", arg13_1: "f16[768]", arg14_1: "f16[3072, 768]", arg15_1: "f16[3072]", arg16_1: "f16[768, 3072]", arg17_1: "f16[768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:215 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        convert_element_type: "f32[4, 512, 768]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[4, 512, 1]" = var_mean[0]
        getitem_1: "f32[4, 512, 1]" = var_mean[1];  var_mean = None
        sub: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul, arg0_1);  mul = arg0_1 = None
        add_1: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_1, arg1_1);  mul_1 = arg1_1 = None
        convert_element_type_1: "f16[4, 512, 768]" = torch.ops.prims.convert_element_type.default(add_1, torch.float16);  add_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:151 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view: "f16[2048, 768]" = torch.ops.aten.reshape.default(convert_element_type_1, [2048, 768])
        permute: "f16[768, 768]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        addmm: "f16[2048, 768]" = torch.ops.aten.addmm.default(arg4_1, view, permute);  arg4_1 = view = permute = None
        view_1: "f16[4, 512, 768]" = torch.ops.aten.reshape.default(addmm, [4, 512, 768]);  addmm = None
        mul_2: "f16[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_1, 0.125);  view_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:152 in forward, code: query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_2: "f16[4, 512, 12, 64]" = torch.ops.aten.reshape.default(mul_2, [4, -1, 12, 64]);  mul_2 = None
        permute_1: "f16[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:154 in forward, code: key_states = self.k_proj(hidden_states)
        view_3: "f16[2048, 768]" = torch.ops.aten.reshape.default(convert_element_type_1, [2048, 768])
        permute_2: "f16[768, 768]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        addmm_1: "f16[2048, 768]" = torch.ops.aten.addmm.default(arg6_1, view_3, permute_2);  arg6_1 = view_3 = permute_2 = None
        view_4: "f16[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_1, [4, 512, 768]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:156 in forward, code: key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_7: "f16[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_4, [4, -1, 12, 64]);  view_4 = None
        permute_4: "f16[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:155 in forward, code: value_states = self.v_proj(hidden_states)
        view_5: "f16[2048, 768]" = torch.ops.aten.reshape.default(convert_element_type_1, [2048, 768]);  convert_element_type_1 = None
        permute_3: "f16[768, 768]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm_2: "f16[2048, 768]" = torch.ops.aten.addmm.default(arg8_1, view_5, permute_3);  arg8_1 = view_5 = permute_3 = None
        view_6: "f16[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_2, [4, 512, 768]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:157 in forward, code: value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        view_8: "f16[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_6, [4, -1, 12, 64]);  view_6 = None
        permute_5: "f16[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_1: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f16[4, 1, 512, 512]" = torch.ops.aten.where.self(arg9_1, full_default_1, full_default);  arg9_1 = full_default_1 = full_default = None
        _scaled_dot_product_cudnn_attention = torch.ops.aten._scaled_dot_product_cudnn_attention.default(permute_1, permute_4, permute_5, where, False, scale = 1.0);  permute_1 = permute_4 = permute_5 = where = None
        getitem_2: "f16[4, 12, 512, 64]" = _scaled_dot_product_cudnn_attention[0];  _scaled_dot_product_cudnn_attention = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[2]" = torch.ops.prims.inductor_seeds.default(2, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:225 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_1: "f32[4, 512, 768]" = torch.ops.prims.inductor_random.default([4, 512, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        convert_element_type_default_1: "f16[4, 512, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_default_1, torch.float16);  inductor_random_default_1 = None
        gt: "b8[4, 512, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_default_1, 0.1);  convert_element_type_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_6: "f16[4, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3]);  getitem_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:178 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        view_9: "f16[4, 512, 768]" = torch.ops.aten.reshape.default(permute_6, [4, 512, -1]);  permute_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:179 in forward, code: attn_output = self.out_proj(attn_output)
        view_10: "f16[2048, 768]" = torch.ops.aten.reshape.default(view_9, [2048, 768]);  view_9 = None
        permute_7: "f16[768, 768]" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        addmm_3: "f16[2048, 768]" = torch.ops.aten.addmm.default(arg11_1, view_10, permute_7);  arg11_1 = view_10 = permute_7 = None
        view_11: "f16[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_3, [4, 512, 768]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:225 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        mul_3: "f16[4, 512, 768]" = torch.ops.aten.mul.Tensor(gt, view_11);  gt = view_11 = None
        mul_4: "f16[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_3, 1.1111111111111112);  mul_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:226 in forward, code: hidden_states = residual + hidden_states
        add_2: "f16[4, 512, 768]" = torch.ops.aten.add.Tensor(arg2_1, mul_4);  arg2_1 = mul_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:234 in forward, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        view_12: "f16[2048, 768]" = torch.ops.aten.reshape.default(add_2, [-1, 768]);  add_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:239 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_14: "f32[2048, 768]" = torch.ops.prims.convert_element_type.default(view_12, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_14, [1], correction = 0, keepdim = True)
        getitem_11: "f32[2048, 1]" = var_mean_1[0]
        getitem_12: "f32[2048, 1]" = var_mean_1[1];  var_mean_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:245 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1);  inductor_seeds_default = None
        inductor_random_default: "f32[2048, 768]" = torch.ops.prims.inductor_random.default([2048, 768], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        convert_element_type_default: "f16[2048, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_default, torch.float16);  inductor_random_default = None
        gt_1: "b8[2048, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_default, 0.1);  convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:239 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        sub_1: "f32[2048, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_14, getitem_12);  convert_element_type_14 = getitem_12 = None
        add_3: "f32[2048, 1]" = torch.ops.aten.add.Tensor(getitem_11, 1e-05);  getitem_11 = None
        rsqrt_1: "f32[2048, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        mul_5: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        mul_6: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(mul_5, arg12_1);  mul_5 = arg12_1 = None
        add_4: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mul_6, arg13_1);  mul_6 = arg13_1 = None
        convert_element_type_15: "f16[2048, 768]" = torch.ops.prims.convert_element_type.default(add_4, torch.float16);  add_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:241 in forward, code: hidden_states = self.fc1(hidden_states)
        permute_8: "f16[768, 3072]" = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
        addmm_4: "f16[2048, 3072]" = torch.ops.aten.addmm.default(arg15_1, convert_element_type_15, permute_8);  arg15_1 = convert_element_type_15 = permute_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:242 in forward, code: hidden_states = self.activation_fn(hidden_states)
        relu: "f16[2048, 3072]" = torch.ops.aten.relu.default(addmm_4);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:244 in forward, code: hidden_states = self.fc2(hidden_states)
        permute_9: "f16[3072, 768]" = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
        addmm_5: "f16[2048, 768]" = torch.ops.aten.addmm.default(arg17_1, relu, permute_9);  arg17_1 = relu = permute_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:245 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        mul_7: "f16[2048, 768]" = torch.ops.aten.mul.Tensor(gt_1, addmm_5);  gt_1 = addmm_5 = None
        mul_8: "f16[2048, 768]" = torch.ops.aten.mul.Tensor(mul_7, 1.1111111111111112);  mul_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:247 in forward, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
        add_5: "f16[2048, 768]" = torch.ops.aten.add.Tensor(view_12, mul_8);  view_12 = mul_8 = None
        view_13: "f16[4, 512, 768]" = torch.ops.aten.reshape.default(add_5, [4, 512, 768]);  add_5 = None
        return (view_13,)

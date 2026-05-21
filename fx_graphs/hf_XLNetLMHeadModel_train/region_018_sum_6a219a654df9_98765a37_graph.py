class GraphModule(torch.nn.Module):
    def forward(self, bmm_195: "f32[256, 512, 512]", gt_94: "b8[16, 16, 512, 512]", div_24: "f32[16, 16, 512, 512]", iota_2: "i64[512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        reshape_default: "f32[16, 16, 512, 512, 1]" = torch.ops.aten.reshape.default(bmm_195, _shape_param_0);  bmm_195 = _shape_param_0 = None
        squeeze_dim: "f32[16, 16, 512, 512]" = torch.ops.aten.squeeze.dim(reshape_default, 4);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:132 in rel_attn_core, code: attn_prob = self.dropout(attn_prob)
        convert_element_type_default: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_94, torch.float32);  gt_94 = None
        mul_tensor: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(squeeze_dim, mul_tensor);  squeeze_dim = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        mul_tensor_2: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_1, div_24);  mul_tensor_1 = None
        sum_dim_int_list: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [3], True)
        neg_default: "f32[16, 16, 512, 512]" = torch.ops.aten.neg.default(div_24);  div_24 = None
        fma_default: "f32[16, 16, 512, 512]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor_2);  neg_default = sum_dim_int_list = mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        mul_tensor_3: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(fma_default, 0.125);  fma_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        full_default: "f32[16, 16, 512, 1023]" = torch.ops.aten.full.default([16, 16, 512, 1023], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[16, 16, 512, 1023]" = torch.ops.aten.index_put.default(full_default, [None, None, None, iota_2], mul_tensor_3, True);  full_default = iota_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        reshape_default_1: "f32[16, 16, 1023, 512]" = torch.ops.aten.reshape.default(index_put_default, _shape_param_1);  index_put_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        full_default_1: "f32[16, 16, 1024, 512]" = torch.ops.aten.full.default([16, 16, 1024, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default: "f32[16, 16, 1024, 512]" = torch.ops.aten.slice_scatter.default(full_default_1, reshape_default_1, 2, 1, 9223372036854775807);  full_default_1 = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        reshape_default_2: "f32[16, 16, 512, 1024]" = torch.ops.aten.reshape.default(slice_scatter_default, _shape_param_2);  slice_scatter_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        reshape_default_3: "f32[16, 16, 512, 1024, 1]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        permute_default: "f32[16, 16, 512, 1, 1024]" = torch.ops.aten.permute.default(reshape_default_3, [0, 1, 2, 4, 3]);  reshape_default_3 = None
        reshape_default_4: "f32[256, 512, 1024]" = torch.ops.aten.reshape.default(permute_default, _shape_param_4);  permute_default = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        reshape_default_5: "f32[16, 16, 512, 512, 1]" = torch.ops.aten.reshape.default(mul_tensor_3, _shape_param_5);  mul_tensor_3 = _shape_param_5 = None
        permute_default_1: "f32[16, 16, 512, 1, 512]" = torch.ops.aten.permute.default(reshape_default_5, [0, 1, 2, 4, 3]);  reshape_default_5 = None
        reshape_default_6: "f32[256, 512, 512]" = torch.ops.aten.reshape.default(permute_default_1, _shape_param_6);  permute_default_1 = _shape_param_6 = None
        return (reshape_default_4, reshape_default_6)

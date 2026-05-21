class GraphModule(torch.nn.Module):
    def forward(self, bmm_524: "f32[256, 512, 64]", bmm_527: "f32[256, 512, 64]", bmm_528: "f32[256, 64, 512]", bmm_529: "f32[256, 512, 64]", primals_20: "f32[1024, 16, 64]", primals_19: "f32[1024, 16, 64]", primals_18: "f32[1024, 16, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        reshape_default: "f32[16, 16, 512, 64, 1]" = torch.ops.aten.reshape.default(bmm_524, _shape_param_0);  bmm_524 = _shape_param_0 = None
        permute_default: "f32[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(reshape_default, [4, 0, 1, 3, 2]);  reshape_default = None
        permute_default_1: "f32[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_default, [4, 1, 2, 3, 0]);  permute_default = None
        squeeze_dim: "f32[512, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_default_1, 4);  permute_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        reshape_default_1: "f32[16, 16, 512, 64, 1]" = torch.ops.aten.reshape.default(bmm_527, _shape_param_1);  bmm_527 = _shape_param_1 = None
        permute_default_2: "f32[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 2, 4, 3]);  reshape_default_1 = None
        permute_default_3: "f32[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_default_2, [2, 0, 1, 4, 3]);  permute_default_2 = None
        squeeze_dim_1: "f32[512, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_default_3, 4);  permute_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        reshape_default_2: "f32[16, 16, 64, 512, 1]" = torch.ops.aten.reshape.default(bmm_528, _shape_param_2);  bmm_528 = _shape_param_2 = None
        permute_default_4: "f32[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(reshape_default_2, [0, 1, 4, 3, 2]);  reshape_default_2 = None
        reshape_default_3: "f32[16, 16, 512, 64, 1]" = torch.ops.aten.reshape.default(bmm_529, _shape_param_3);  bmm_529 = _shape_param_3 = None
        permute_default_5: "f32[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(reshape_default_3, [0, 1, 2, 4, 3]);  reshape_default_3 = None
        permute_default_6: "f32[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_default_4, [3, 0, 1, 4, 2]);  permute_default_4 = None
        squeeze_dim_2: "f32[512, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_default_6, 4);  permute_default_6 = None
        permute_default_7: "f32[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_default_5, [2, 0, 1, 4, 3]);  permute_default_5 = None
        squeeze_dim_3: "f32[512, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_default_7, 4);  permute_default_7 = None
        add_tensor: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(squeeze_dim_1, squeeze_dim_3);  squeeze_dim_1 = squeeze_dim_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        reshape_default_4: "f32[512, 16, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_dim, _shape_param_4);  squeeze_dim = _shape_param_4 = None
        permute_default_8: "f32[512, 16, 1, 16, 64]" = torch.ops.aten.permute.default(reshape_default_4, [0, 1, 4, 2, 3]);  reshape_default_4 = None
        clone_default: "f32[512, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_default_8, memory_format = torch.contiguous_format);  permute_default_8 = None
        reshape_default_5: "f32[1, 8192, 1024]" = torch.ops.aten.reshape.default(clone_default, _shape_param_5);  clone_default = _shape_param_5 = None
        squeeze_dim_4: "f32[8192, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_5, 0);  reshape_default_5 = None
        unsqueeze_default: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_20, 3);  primals_20 = None
        unsqueeze_default_1: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 4);  unsqueeze_default = None
        reshape_default_6: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_1, _shape_param_6);  unsqueeze_default_1 = _shape_param_6 = None
        permute_default_9: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(reshape_default_6, [0, 2, 1]);  reshape_default_6 = None
        squeeze_dim_5: "f32[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_default_9, 0);  permute_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        reshape_default_7: "f32[512, 16, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_dim_2, _shape_param_7);  squeeze_dim_2 = _shape_param_7 = None
        permute_default_10: "f32[512, 16, 1, 16, 64]" = torch.ops.aten.permute.default(reshape_default_7, [0, 1, 4, 2, 3]);  reshape_default_7 = None
        clone_default_1: "f32[512, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_default_10, memory_format = torch.contiguous_format);  permute_default_10 = None
        reshape_default_8: "f32[1, 8192, 1024]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_8);  clone_default_1 = _shape_param_8 = None
        squeeze_dim_6: "f32[8192, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_8, 0);  reshape_default_8 = None
        unsqueeze_default_2: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_19, 3);  primals_19 = None
        unsqueeze_default_3: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, 4);  unsqueeze_default_2 = None
        reshape_default_9: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_3, _shape_param_9);  unsqueeze_default_3 = _shape_param_9 = None
        permute_default_11: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(reshape_default_9, [0, 2, 1]);  reshape_default_9 = None
        squeeze_dim_7: "f32[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_default_11, 0);  permute_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        reshape_default_10: "f32[512, 16, 16, 64, 1]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_10);  add_tensor = _shape_param_10 = None
        permute_default_12: "f32[512, 16, 1, 16, 64]" = torch.ops.aten.permute.default(reshape_default_10, [0, 1, 4, 2, 3]);  reshape_default_10 = None
        clone_default_2: "f32[512, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_default_12, memory_format = torch.contiguous_format);  permute_default_12 = None
        reshape_default_11: "f32[1, 8192, 1024]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_11);  clone_default_2 = _shape_param_11 = None
        squeeze_dim_8: "f32[8192, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_11, 0);  reshape_default_11 = None
        unsqueeze_default_4: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(primals_18, 3);  primals_18 = None
        unsqueeze_default_5: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 4);  unsqueeze_default_4 = None
        reshape_default_12: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_5, _shape_param_12);  unsqueeze_default_5 = _shape_param_12 = None
        permute_default_13: "f32[1, 1024, 1024]" = torch.ops.aten.permute.default(reshape_default_12, [0, 2, 1]);  reshape_default_12 = None
        squeeze_dim_9: "f32[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_default_13, 0);  permute_default_13 = None
        return (squeeze_dim_4, squeeze_dim_5, squeeze_dim_6, squeeze_dim_7, squeeze_dim_8, squeeze_dim_9)

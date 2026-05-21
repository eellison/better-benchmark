class GraphModule(torch.nn.Module):
    def forward(self, mm_default_7: "f32[8192, 1024]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_default: "f32[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_7, 0);  mm_default_7 = None
        reshape_default: "f32[512, 16, 64, 16, 1]" = torch.ops.aten.reshape.default(unsqueeze_default, _shape_param_0);  unsqueeze_default = _shape_param_0 = None
        permute_default: "f32[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(reshape_default, [0, 1, 4, 2, 3]);  reshape_default = None
        permute_default_1: "f32[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_default, [0, 1, 4, 3, 2]);  permute_default = None
        squeeze_dim: "f32[512, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_default_1, 4);  permute_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        reshape_default_1: "f32[512, 16, 16, 64, 1]" = torch.ops.aten.reshape.default(squeeze_dim, _shape_param_1);  squeeze_dim = _shape_param_1 = None
        permute_default_2: "f32[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(reshape_default_1, [1, 2, 0, 4, 3]);  reshape_default_1 = None
        clone_default: "f32[16, 16, 512, 1, 64]" = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        reshape_default_2: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(clone_default, _shape_param_2);  clone_default = _shape_param_2 = None
        return reshape_default_2

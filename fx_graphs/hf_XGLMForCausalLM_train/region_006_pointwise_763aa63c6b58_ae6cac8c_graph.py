class GraphModule(torch.nn.Module):
    def forward(self, bmm_1: "f32[512, 128, 64]", primals_11: "f32[1024, 1024]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:236 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        reshape_default: "f32[32, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_1, _shape_param_0);  bmm_1 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:237 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_default: "f32[32, 128, 16, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:241 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_default: "f32[32, 128, 16, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_1: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:243 in forward, code: attn_output = self.out_proj(attn_output)
        reshape_default_2: "f32[4096, 1024]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
        permute_default_1: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
        return (reshape_default_2, permute_default_1)

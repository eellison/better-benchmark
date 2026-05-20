class GraphModule(torch.nn.Module):
    def forward(self, getitem_100: "f32[16, 16, 512, 64]", arg382_1: "f32[1024, 1024]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_default: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_100, [0, 2, 1, 3]);  getitem_100 = None
        clone_default: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        reshape_default: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_default, _shape_param_0);  clone_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default_1: "f32[8192, 1024]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default_1: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg382_1, [1, 0]);  arg382_1 = None
        return (reshape_default_1, permute_default_1)

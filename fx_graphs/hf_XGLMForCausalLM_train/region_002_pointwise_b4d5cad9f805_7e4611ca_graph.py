class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[1, 128]", arg1_1: "f32[2050, 1024]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:96 in forward, code: position_ids = position_ids + self.offset
        add_tensor: "i64[1, 128]" = torch.ops.aten.add.Tensor(arg0_1, 2);  arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xglm/modeling_xglm.py:102 in forward, code: return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, self.weights.shape[-1]).detach()
        reshape_default: "i64[128]" = torch.ops.aten.reshape.default(add_tensor, [-1]);  add_tensor = None
        index_tensor: "f32[128, 1024]" = torch.ops.aten.index.Tensor(arg1_1, [reshape_default]);  arg1_1 = reshape_default = None
        reshape_default_1: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(index_tensor, _shape_param_0);  index_tensor = _shape_param_0 = None
        return reshape_default_1

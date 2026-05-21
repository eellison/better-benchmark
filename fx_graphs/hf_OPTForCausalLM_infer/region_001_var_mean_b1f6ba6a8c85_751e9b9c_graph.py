class GraphModule(torch.nn.Module):
    def forward(self, view_167: "f32[8192, 768]", addmm_71: "f32[8192, 768]", arg195_1: "f32[768]", arg196_1: "f32[768]", arg1_1: "f32[50272, 768]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:247 in forward, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
        add_tensor: "f32[8192, 768]" = torch.ops.aten.add.Tensor(view_167, addmm_71);  view_167 = addmm_71 = None
        reshape_default: "f32[4, 2048, 768]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_0);  add_tensor = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:388 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_correction = torch.ops.aten.var_mean.correction(reshape_default, [2], correction = 0, keepdim = True)
        getitem: "f32[4, 2048, 1]" = var_mean_correction[0]
        getitem_1: "f32[4, 2048, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[4, 2048, 768]" = torch.ops.aten.sub.Tensor(reshape_default, getitem_1);  reshape_default = getitem_1 = None
        add_tensor_1: "f32[4, 2048, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[4, 2048, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[4, 2048, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[4, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg195_1);  mul_tensor = arg195_1 = None
        add_tensor_2: "f32[4, 2048, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg196_1);  mul_tensor_1 = arg196_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:512 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :]).contiguous()
        reshape_default_1: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_1);  add_tensor_2 = _shape_param_1 = None
        permute_default: "f32[768, 50272]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        return (reshape_default_1, permute_default)

class GraphModule(torch.nn.Module):
    def forward(self, addmm_69: "f32[8192, 768]", view_154: "f32[4, 2048, 768]", arg189_1: "f32[768]", arg190_1: "f32[768]", arg191_1: "f32[3072, 768]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:179 in forward, code: attn_output = self.out_proj(attn_output)
        reshape_default: "f32[4, 2048, 768]" = torch.ops.aten.reshape.default(addmm_69, _shape_param_0);  addmm_69 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:226 in forward, code: hidden_states = residual + hidden_states
        add_tensor: "f32[4, 2048, 768]" = torch.ops.aten.add.Tensor(view_154, reshape_default);  view_154 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:234 in forward, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        reshape_default_1: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_1);  add_tensor = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:239 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_correction = torch.ops.aten.var_mean.correction(reshape_default_1, [1], correction = 0, keepdim = True)
        getitem: "f32[8192, 1]" = var_mean_correction[0]
        getitem_1: "f32[8192, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[8192, 768]" = torch.ops.aten.sub.Tensor(reshape_default_1, getitem_1);  reshape_default_1 = getitem_1 = None
        add_tensor_1: "f32[8192, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[8192, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[8192, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[8192, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg189_1);  mul_tensor = arg189_1 = None
        add_tensor_2: "f32[8192, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg190_1);  mul_tensor_1 = arg190_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:241 in forward, code: hidden_states = self.fc1(hidden_states)
        permute_default: "f32[768, 3072]" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        return (add_tensor_2, permute_default)

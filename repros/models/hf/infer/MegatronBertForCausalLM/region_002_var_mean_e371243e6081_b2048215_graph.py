class GraphModule(torch.nn.Module):
    def forward(self, addmm_143: "f32[8192, 1024]", add_189: "f32[16, 512, 1024]", arg390_1: "f32[1024]", arg391_1: "f32[1024]", arg392_1: "f32[1024, 1024]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_143, _shape_param_0);  addmm_143 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_tensor: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_189, reshape_default);  add_189 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:392 in forward, code: hidden_states = self.ln(hidden_states)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[16, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[16, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        add_tensor_1: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, arg390_1);  mul_tensor = arg390_1 = None
        add_tensor_2: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg391_1);  mul_tensor_1 = arg391_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:446 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default_1: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_1);  add_tensor_2 = _shape_param_1 = None
        permute_default: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg392_1, [1, 0]);  arg392_1 = None
        return (reshape_default_1, permute_default)

class GraphModule(torch.nn.Module):
    def forward(self, addmm_361: "f32[32768, 512]", primals_1116: "f32[512]", primals_1117: "f32[512]", primals_3: "f32[30522, 128]", primals_1118: "f32[384, 30522]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:489 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_361, _shape_param_0);  addmm_361 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:490 in forward, code: hidden_states = self.transform_act_fn(hidden_states)
        relu_default: "f32[256, 128, 512]" = torch.ops.aten.relu.default(reshape_default);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:491 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        var_mean_correction = torch.ops.aten.var_mean.correction(relu_default, [2], correction = 0, keepdim = True)
        getitem: "f32[256, 128, 1]" = var_mean_correction[0]
        getitem_1: "f32[256, 128, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[256, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[256, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[256, 128, 512]" = torch.ops.aten.sub.Tensor(relu_default, getitem_1);  relu_default = getitem_1 = None
        mul_tensor: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, primals_1116);  mul_tensor = primals_1116 = None
        add_tensor_1: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_tensor_1, primals_1117);  mul_tensor_1 = primals_1117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:507 in forward, code: hidden_states = hidden_states.matmul(torch.cat([self.decoder.weight.t(), self.dense.weight], dim=0))
        permute_default: "f32[128, 30522]" = torch.ops.aten.permute.default(primals_3, [1, 0]);  primals_3 = None
        cat_default: "f32[512, 30522]" = torch.ops.aten.cat.default([permute_default, primals_1118]);  permute_default = primals_1118 = None
        reshape_default_1: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_1);  add_tensor_1 = _shape_param_1 = None
        constant_pad_nd_default: "f32[512, 30524]" = torch.ops.aten.constant_pad_nd.default(cat_default, [0, 2, 0, 0]);  cat_default = None
        return (reshape_default_1, constant_pad_nd_default)

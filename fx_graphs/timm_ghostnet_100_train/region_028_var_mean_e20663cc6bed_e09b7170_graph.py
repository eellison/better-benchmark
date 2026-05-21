class GraphModule(torch.nn.Module):
    def forward(self, convolution_47: "f32[512, 92, 14, 14]", primals_272: "f32[92]", primals_273: "f32[92]", relu_19: "f32[512, 92, 14, 14]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_47, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 92, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 92, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 92, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 92, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[512, 92, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_47, getitem_1);  convolution_47 = getitem_1 = None
        mul_tensor: "f32[512, 92, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(primals_272, -1);  primals_272 = None
        unsqueeze_default_1: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[512, 92, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(primals_273, -1);  primals_273 = None
        unsqueeze_default_3: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[512, 92, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        relu_default: "f32[512, 92, 14, 14]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        cat_default: "f32[512, 184, 14, 14]" = torch.ops.aten.cat.default([relu_19, relu_default], 1);  relu_19 = relu_default = None
        return cat_default

class GraphModule(torch.nn.Module):
    def forward(self, convolution_13: "f32[128, 160, 28, 28]", add_21: "f32[128, 160, 28, 28]", arg49_1: "f32[160]", arg50_1: "f32[160]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        add_tensor: "f32[128, 160, 28, 28]" = torch.ops.aten.add.Tensor(convolution_13, add_21);  convolution_13 = add_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_default: "f32[128, 28, 28, 160]" = torch.ops.aten.permute.default(add_tensor, [0, 2, 3, 1]);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        var_mean_correction = torch.ops.aten.var_mean.correction(permute_default, [3], correction = 0, keepdim = True)
        getitem: "f32[128, 28, 28, 1]" = var_mean_correction[0]
        getitem_1: "f32[128, 28, 28, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[128, 28, 28, 160]" = torch.ops.aten.sub.Tensor(permute_default, getitem_1);  permute_default = getitem_1 = None
        add_tensor_1: "f32[128, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt_default: "f32[128, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[128, 28, 28, 160]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[128, 28, 28, 160]" = torch.ops.aten.mul.Tensor(mul_tensor, arg49_1);  mul_tensor = arg49_1 = None
        add_tensor_2: "f32[128, 28, 28, 160]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg50_1);  mul_tensor_1 = arg50_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_default_1: "f32[128, 160, 28, 28]" = torch.ops.aten.permute.default(add_tensor_2, [0, 3, 1, 2]);  add_tensor_2 = None
        return permute_default_1

class GraphModule(torch.nn.Module):
    def forward(self, addmm_7: "f32[401408, 128]", view_48: "f32[128, 3136, 128]", arg34_1: "f32[512]", arg35_1: "f32[512]", arg36_1: "f32[256, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default: "f32[128, 3136, 128]" = torch.ops.aten.reshape.default(addmm_7, _shape_param_0);  addmm_7 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_tensor: "f32[128, 3136, 128]" = torch.ops.aten.add.Tensor(view_48, reshape_default);  view_48 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        reshape_default_1: "f32[128, 56, 56, 128]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_1);  add_tensor = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:539 in forward, code: x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        reshape_default_2: "f32[128, 28, 2, 28, 2, 128]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
        permute_default: "f32[128, 28, 28, 2, 2, 128]" = torch.ops.aten.permute.default(reshape_default_2, [0, 1, 3, 4, 2, 5]);  reshape_default_2 = None
        clone_default: "f32[128, 28, 28, 2, 2, 128]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_3: "f32[128, 28, 28, 512]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:540 in forward, code: x = self.norm(x)
        var_mean_correction = torch.ops.aten.var_mean.correction(reshape_default_3, [3], correction = 0, keepdim = True)
        getitem: "f32[128, 28, 28, 1]" = var_mean_correction[0]
        getitem_1: "f32[128, 28, 28, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[128, 28, 28, 512]" = torch.ops.aten.sub.Tensor(reshape_default_3, getitem_1);  reshape_default_3 = getitem_1 = None
        add_tensor_1: "f32[128, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[128, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[128, 28, 28, 512]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[128, 28, 28, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, arg34_1);  mul_tensor = arg34_1 = None
        add_tensor_2: "f32[128, 28, 28, 512]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg35_1);  mul_tensor_1 = arg35_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:541 in forward, code: x = self.reduction(x)
        reshape_default_4: "f32[100352, 512]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_4);  add_tensor_2 = _shape_param_4 = None
        permute_default_1: "f32[512, 256]" = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
        return (reshape_default_4, permute_default_1)

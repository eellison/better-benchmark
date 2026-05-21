class GraphModule(torch.nn.Module):
    def forward(self, mm_50: "f32[32768, 384]", _shape_param_0, addmm_10: "f32[32768, 384]", _shape_param_1, _shape_param_2, primals_176: "f32[384, 192]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default: "f32[512, 64, 384]" = torch.ops.aten.reshape.default(mm_50, _shape_param_0);  mm_50 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        reshape_default_1: "f32[512, 64, 384]" = torch.ops.aten.reshape.default(addmm_10, _shape_param_1);  addmm_10 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        neg_default: "f32[512, 64, 384]" = torch.ops.aten.neg.default(reshape_default_1)
        exp_default: "f32[512, 64, 384]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[512, 64, 384]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        reciprocal_default: "f32[512, 64, 384]" = torch.ops.aten.reciprocal.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[512, 64, 384]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        mul_tensor_1: "f32[512, 64, 384]" = torch.ops.aten.mul.Tensor(reshape_default, mul_tensor);  reshape_default = None
        sub_tensor: "f32[512, 64, 384]" = torch.ops.aten.sub.Tensor(1, mul_tensor);  mul_tensor = None
        mul_tensor_2: "f32[512, 64, 384]" = torch.ops.aten.mul.Tensor(reshape_default_1, sub_tensor);  reshape_default_1 = sub_tensor = None
        add_tensor_1: "f32[512, 64, 384]" = torch.ops.aten.add.Tensor(mul_tensor_2, 1);  mul_tensor_2 = None
        mul_tensor_3: "f32[512, 64, 384]" = torch.ops.aten.mul.Tensor(mul_tensor_1, add_tensor_1);  mul_tensor_1 = add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        reshape_default_2: "f32[32768, 384]" = torch.ops.aten.reshape.default(mul_tensor_3, _shape_param_2);  mul_tensor_3 = _shape_param_2 = None
        permute_default: "f32[192, 384]" = torch.ops.aten.permute.default(primals_176, [1, 0]);  primals_176 = None
        permute_default_1: "f32[384, 192]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_2, permute_default_1)

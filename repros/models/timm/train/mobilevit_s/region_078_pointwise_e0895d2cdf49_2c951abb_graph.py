class GraphModule(torch.nn.Module):
    def forward(self, mm_18: "f32[8192, 480]", addmm_26: "f32[8192, 480]", primals_263: "f32[480, 240]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default: "f32[512, 16, 480]" = torch.ops.aten.reshape.default(mm_18, _shape_param_0);  mm_18 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        reshape_default_1: "f32[512, 16, 480]" = torch.ops.aten.reshape.default(addmm_26, _shape_param_1);  addmm_26 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        neg_default: "f32[512, 16, 480]" = torch.ops.aten.neg.default(reshape_default_1)
        exp_default: "f32[512, 16, 480]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[512, 16, 480]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        reciprocal_default: "f32[512, 16, 480]" = torch.ops.aten.reciprocal.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[512, 16, 480]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        mul_tensor_1: "f32[512, 16, 480]" = torch.ops.aten.mul.Tensor(reshape_default, mul_tensor);  reshape_default = None
        sub_tensor: "f32[512, 16, 480]" = torch.ops.aten.sub.Tensor(1, mul_tensor);  mul_tensor = None
        mul_tensor_2: "f32[512, 16, 480]" = torch.ops.aten.mul.Tensor(reshape_default_1, sub_tensor);  reshape_default_1 = sub_tensor = None
        add_tensor_1: "f32[512, 16, 480]" = torch.ops.aten.add.Tensor(mul_tensor_2, 1);  mul_tensor_2 = None
        mul_tensor_3: "f32[512, 16, 480]" = torch.ops.aten.mul.Tensor(mul_tensor_1, add_tensor_1);  mul_tensor_1 = add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        reshape_default_2: "f32[8192, 480]" = torch.ops.aten.reshape.default(mul_tensor_3, _shape_param_2);  mul_tensor_3 = _shape_param_2 = None
        permute_default: "f32[240, 480]" = torch.ops.aten.permute.default(primals_263, [1, 0]);  primals_263 = None
        permute_default_1: "f32[480, 240]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_2, permute_default_1)

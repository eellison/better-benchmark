class GraphModule(torch.nn.Module):
    def forward(self, mm_52: "f32[32768, 192]", primals_174: "f32[192]", mul_173: "f32[512, 64, 192]", div_49: "f32[512, 64, 1]", add_296: "f32[512, 64, 192]", primals_172: "f32[192, 192]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        reshape_default: "f32[512, 64, 192]" = torch.ops.aten.reshape.default(mm_52, _shape_param_0);  mm_52 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        mul_tensor: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(reshape_default, primals_174);  reshape_default = primals_174 = None
        mul_tensor_1: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(mul_tensor, 192)
        sum_dim_int_list: "f32[512, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_173);  mul_tensor = None
        sum_dim_int_list_1: "f32[512, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(mul_173, sum_dim_int_list_1);  mul_173 = sum_dim_int_list_1 = None
        sub_tensor: "f32[512, 64, 192]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[512, 64, 192]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(div_49, sub_tensor_1);  div_49 = sub_tensor_1 = None
        add_tensor: "f32[512, 64, 192]" = torch.ops.aten.add.Tensor(add_296, mul_tensor_4);  add_296 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_1: "f32[32768, 192]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_1);  add_tensor = _shape_param_1 = None
        permute_default: "f32[192, 192]" = torch.ops.aten.permute.default(primals_172, [1, 0]);  primals_172 = None
        permute_default_1: "f32[192, 192]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_1, permute_default_1)

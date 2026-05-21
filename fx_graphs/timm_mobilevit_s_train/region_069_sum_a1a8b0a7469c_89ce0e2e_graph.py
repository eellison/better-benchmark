class GraphModule(torch.nn.Module):
    def forward(self, getitem_208: "f32[128, 192, 16, 16]", _shape_param_0, _shape_param_1, _shape_param_2, primals_216: "f32[192]", mul_187: "f32[512, 64, 192]", div_42: "f32[512, 64, 1]", _shape_param_3, primals_214: "f32[192, 384]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:271 in forward, code: x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
        clone_default: "f32[128, 192, 16, 16]" = torch.ops.aten.clone.default(getitem_208, memory_format = torch.contiguous_format);  getitem_208 = None
        reshape_default: "f32[196608, 2, 8, 2]" = torch.ops.aten.reshape.default(clone_default, _shape_param_0);  clone_default = _shape_param_0 = None
        permute_default: "f32[196608, 8, 2, 2]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:269 in forward, code: x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
        clone_default_1: "f32[196608, 8, 2, 2]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_1: "f32[128, 192, 64, 4]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_1);  clone_default_1 = _shape_param_1 = None
        permute_default_1: "f32[128, 4, 64, 192]" = torch.ops.aten.permute.default(reshape_default_1, [0, 3, 2, 1]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:268 in forward, code: x = x.contiguous().view(B, self.patch_area, num_patches, -1)
        clone_default_2: "f32[128, 4, 64, 192]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_2: "f32[512, 64, 192]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_2);  clone_default_2 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:264 in forward, code: x = self.norm(x)
        mul_tensor: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(reshape_default_2, primals_216);  reshape_default_2 = primals_216 = None
        mul_tensor_1: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(mul_tensor, 192)
        sum_dim_int_list: "f32[512, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_187);  mul_tensor = None
        sum_dim_int_list_1: "f32[512, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(mul_187, sum_dim_int_list_1);  mul_187 = sum_dim_int_list_1 = None
        sub_tensor: "f32[512, 64, 192]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[512, 64, 192]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[512, 64, 192]" = torch.ops.aten.mul.Tensor(div_42, sub_tensor_1);  div_42 = sub_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default_3: "f32[32768, 192]" = torch.ops.aten.reshape.default(mul_tensor_4, _shape_param_3);  mul_tensor_4 = _shape_param_3 = None
        permute_default_2: "f32[384, 192]" = torch.ops.aten.permute.default(primals_214, [1, 0]);  primals_214 = None
        permute_default_3: "f32[192, 384]" = torch.ops.aten.permute.default(permute_default_2, [1, 0]);  permute_default_2 = None
        return (reshape_default_3, permute_default_3)

class GraphModule(torch.nn.Module):
    def forward(self, convolution_16: "f32[128, 1280, 14, 14]", pow_10: "f32[128, 1280, 1, 1]", add_33: "f32[128, 1, 1, 1]", primals_61: "f32[1280]", getitem_122: "f32[128, 1280, 14, 14]", full_default: "f32[]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_tensor: "f32[128, 1280, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_16, 0.5)
        mul_tensor_1: "f32[128, 1280, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_16, 0.7071067811865476)
        erf_default: "f32[128, 1280, 14, 14]" = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor: "f32[128, 1280, 14, 14]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_2: "f32[128, 1280, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_tensor: "f32[128, 1280, 1, 1]" = torch.ops.aten.div.Tensor(pow_10, add_33)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        reshape_default: "f32[1, 1280, 1, 1]" = torch.ops.aten.reshape.default(primals_61, [1, -1, 1, 1]);  primals_61 = None
        mul_scalar: "f32[1, 1280, 1, 1]" = torch.ops.aten.mul.Scalar(reshape_default, 1);  reshape_default = None
        mul_tensor_3: "f32[128, 1280, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_122, mul_scalar);  mul_scalar = None
        mul_tensor_4: "f32[128, 1280, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor_3, mul_tensor_2)
        mul_tensor_5: "f32[128, 1280, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor_3, div_tensor);  mul_tensor_3 = None
        sum_dim_int_list: "f32[128, 1280, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [2, 3], True);  mul_tensor_4 = None
        add_tensor_1: "f32[128, 1280, 14, 14]" = torch.ops.aten.add.Tensor(getitem_122, mul_tensor_5);  getitem_122 = mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_tensor_1: "f32[128, 1280, 1, 1]" = torch.ops.aten.div.Tensor(div_tensor, add_33);  div_tensor = None
        neg_default: "f32[128, 1280, 1, 1]" = torch.ops.aten.neg.default(sum_dim_int_list)
        mul_tensor_6: "f32[128, 1280, 1, 1]" = torch.ops.aten.mul.Tensor(neg_default, div_tensor_1);  neg_default = div_tensor_1 = None
        div_tensor_2: "f32[128, 1280, 1, 1]" = torch.ops.aten.div.Tensor(sum_dim_int_list, add_33);  sum_dim_int_list = add_33 = None
        sum_dim_int_list_1: "f32[128, 1, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [1], True);  mul_tensor_6 = None
        expand_default: "f32[128, 1280, 1, 1]" = torch.ops.aten.expand.default(sum_dim_int_list_1, _shape_param_0);  sum_dim_int_list_1 = _shape_param_0 = None
        div_scalar: "f32[128, 1280, 1, 1]" = torch.ops.aten.div.Scalar(expand_default, 1280);  expand_default = None
        add_tensor_2: "f32[128, 1280, 1, 1]" = torch.ops.aten.add.Tensor(div_tensor_2, div_scalar);  div_tensor_2 = div_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        div_tensor_3: "f32[128, 1280, 14, 14]" = torch.ops.aten.div.Tensor(mul_tensor_2, pow_10);  mul_tensor_2 = None
        eq_scalar: "b8[128, 1280, 1, 1]" = torch.ops.aten.eq.Scalar(pow_10, 0);  pow_10 = None
        where_self: "f32[128, 1280, 14, 14]" = torch.ops.aten.where.self(eq_scalar, full_default, div_tensor_3);  eq_scalar = full_default = div_tensor_3 = None
        clone_default: "f32[128, 1280, 14, 14]" = torch.ops.aten.clone.default(where_self, memory_format = torch.contiguous_format);  where_self = None
        mul_tensor_7: "f32[128, 1280, 14, 14]" = torch.ops.aten.mul.Tensor(add_tensor_2, clone_default);  add_tensor_2 = clone_default = None
        add_tensor_3: "f32[128, 1280, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor_7);  add_tensor_1 = mul_tensor_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_tensor_8: "f32[128, 1280, 14, 14]" = torch.ops.aten.mul.Tensor(add_tensor, 0.5);  add_tensor = None
        mul_tensor_9: "f32[128, 1280, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_16, convolution_16)
        mul_tensor_10: "f32[128, 1280, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor_9, -0.5);  mul_tensor_9 = None
        exp_default: "f32[128, 1280, 14, 14]" = torch.ops.aten.exp.default(mul_tensor_10);  mul_tensor_10 = None
        mul_tensor_11: "f32[128, 1280, 14, 14]" = torch.ops.aten.mul.Tensor(exp_default, 0.3989422804014327);  exp_default = None
        mul_tensor_12: "f32[128, 1280, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_16, mul_tensor_11);  convolution_16 = mul_tensor_11 = None
        add_tensor_4: "f32[128, 1280, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_8, mul_tensor_12);  mul_tensor_8 = mul_tensor_12 = None
        mul_tensor_13: "f32[128, 1280, 14, 14]" = torch.ops.aten.mul.Tensor(add_tensor_3, add_tensor_4);  add_tensor_3 = add_tensor_4 = None
        return mul_tensor_13

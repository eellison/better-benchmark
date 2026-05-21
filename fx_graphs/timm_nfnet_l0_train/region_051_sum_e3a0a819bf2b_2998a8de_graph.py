class GraphModule(torch.nn.Module):
    def forward(self, getitem_261: "f32[128, 1536, 14, 14]", convolution_30: "f32[128, 1536, 1, 1]", convolution_28: "f32[128, 1536, 14, 14]", convolution_24: "f32[128, 1536, 14, 14]", add_192: "f32[128, 1536, 14, 14]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        mul_tensor: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_261, 0.9805806756909201);  getitem_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_default: "f32[128, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_30);  convolution_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_tensor_1: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_28, sigmoid_default)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_tensor_2: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor_1, 2.0);  mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_tensor_3: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 0.2);  mul_tensor_2 = None
        add_tensor: "f32[128, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_3, convolution_24);  mul_tensor_3 = convolution_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        neg_default: "f32[128, 1536, 14, 14]" = torch.ops.aten.neg.default(add_tensor)
        exp_default: "f32[128, 1536, 14, 14]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor_1: "f32[128, 1536, 14, 14]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        reciprocal_default: "f32[128, 1536, 14, 14]" = torch.ops.aten.reciprocal.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor_4: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        mul_tensor_5: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_4);  mul_tensor = None
        sub_tensor: "f32[128, 1536, 14, 14]" = torch.ops.aten.sub.Tensor(1, mul_tensor_4);  mul_tensor_4 = None
        mul_tensor_6: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_tensor, sub_tensor);  add_tensor = sub_tensor = None
        add_tensor_2: "f32[128, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_6, 1);  mul_tensor_6 = None
        mul_tensor_7: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor_5, add_tensor_2);  mul_tensor_5 = add_tensor_2 = None
        add_tensor_3: "f32[128, 1536, 14, 14]" = torch.ops.aten.add.Tensor(add_192, mul_tensor_7);  add_192 = mul_tensor_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_tensor_8: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_tensor_3, 0.2);  add_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_tensor_9: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor_8, 2.0);  mul_tensor_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_tensor_10: "f32[128, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor_9, convolution_28);  mul_tensor_9 = convolution_28 = None
        sum_dim_int_list: "f32[128, 1536, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [2, 3], True);  mul_tensor_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sub_tensor_1: "f32[128, 1536, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_default)
        mul_tensor_11: "f32[128, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_default, sub_tensor_1);  sigmoid_default = sub_tensor_1 = None
        mul_tensor_12: "f32[128, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, mul_tensor_11);  sum_dim_int_list = mul_tensor_11 = None
        return mul_tensor_12

class GraphModule(torch.nn.Module):
    def forward(self, getitem_300: "f32[128, 512, 24, 24]", convolution_17: "f32[128, 512, 1, 1]", convolution_15: "f32[128, 512, 24, 24]", primals_53: "f32[]", convolution_11: "f32[128, 512, 24, 24]", mul_1205: "f32[128, 512, 24, 24]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        mul_tensor: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(getitem_300, 0.9805806756909201);  getitem_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_tensor_1: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.7015043497085571);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_default: "f32[128, 512, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_17);  convolution_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_tensor_2: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_15, sigmoid_default)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_tensor_3: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 2.0);  mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:280 in forward, code: out.mul_(self.skipinit_gain)
        mul_tensor_4: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor_3, primals_53);  mul_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_tensor_5: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor_4, 0.2);  mul_tensor_4 = None
        add_tensor: "f32[128, 512, 24, 24]" = torch.ops.aten.add.Tensor(mul_tensor_5, convolution_11);  mul_tensor_5 = convolution_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_tensor_6: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(add_tensor, 0.7071067811865476)
        erf_default: "f32[128, 512, 24, 24]" = torch.ops.aten.erf.default(mul_tensor_6);  mul_tensor_6 = None
        add_tensor_1: "f32[128, 512, 24, 24]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_7: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(add_tensor_1, 0.5);  add_tensor_1 = None
        mul_tensor_8: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(add_tensor, add_tensor)
        mul_tensor_9: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor_8, -0.5);  mul_tensor_8 = None
        exp_default: "f32[128, 512, 24, 24]" = torch.ops.aten.exp.default(mul_tensor_9);  mul_tensor_9 = None
        mul_tensor_10: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(exp_default, 0.3989422804014327);  exp_default = None
        mul_tensor_11: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(add_tensor, mul_tensor_10);  add_tensor = mul_tensor_10 = None
        add_tensor_2: "f32[128, 512, 24, 24]" = torch.ops.aten.add.Tensor(mul_tensor_7, mul_tensor_11);  mul_tensor_7 = mul_tensor_11 = None
        mul_tensor_12: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor_1, add_tensor_2);  mul_tensor_1 = add_tensor_2 = None
        add_tensor_3: "f32[128, 512, 24, 24]" = torch.ops.aten.add.Tensor(mul_1205, mul_tensor_12);  mul_1205 = mul_tensor_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_tensor_13: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(add_tensor_3, 0.2);  add_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:280 in forward, code: out.mul_(self.skipinit_gain)
        mul_tensor_14: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor_13, primals_53);  mul_tensor_13 = primals_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_tensor_15: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor_14, 2.0);  mul_tensor_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_tensor_16: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor_15, convolution_15);  mul_tensor_15 = convolution_15 = None
        sum_dim_int_list: "f32[128, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [2, 3], True);  mul_tensor_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sub_tensor: "f32[128, 512, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_default)
        mul_tensor_17: "f32[128, 512, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_default, sub_tensor);  sigmoid_default = sub_tensor = None
        mul_tensor_18: "f32[128, 512, 1, 1]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, mul_tensor_17);  sum_dim_int_list = mul_tensor_17 = None
        return mul_tensor_18

class GraphModule(torch.nn.Module):
    def forward(self, convolution_17: "f32[128, 512, 1, 1]", convolution_15: "f32[128, 512, 24, 24]", primals_53: "f32[]", convolution_11: "f32[128, 512, 24, 24]", primals_54: "f32[256, 512, 1, 1]", primals_55: "f32[256, 1, 1, 1]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_default: "f32[128, 512, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_17);  convolution_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_tensor: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_15, sigmoid_default);  convolution_15 = sigmoid_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_tensor_1: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor, 2.0);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:280 in forward, code: out.mul_(self.skipinit_gain)
        mul_tensor_2: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor_1, primals_53);  mul_tensor_1 = primals_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_tensor_3: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 0.2);  mul_tensor_2 = None
        add_tensor: "f32[128, 512, 24, 24]" = torch.ops.aten.add.Tensor(mul_tensor_3, convolution_11);  mul_tensor_3 = convolution_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_tensor_4: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(add_tensor, 0.5)
        mul_tensor_5: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(add_tensor, 0.7071067811865476);  add_tensor = None
        erf_default: "f32[128, 512, 24, 24]" = torch.ops.aten.erf.default(mul_tensor_5);  mul_tensor_5 = None
        add_tensor_1: "f32[128, 512, 24, 24]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_6: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor_4, add_tensor_1);  mul_tensor_4 = add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_tensor_7: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor_6, 1.7015043497085571);  mul_tensor_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        mul_tensor_8: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor_7, 0.9805806756909201);  mul_tensor_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default: "f32[1, 256, 512]" = torch.ops.aten.reshape.default(primals_54, _shape_param_0);  primals_54 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_9: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_55, 0.04419417382415922);  primals_55 = None
        reshape_default_1: "f32[256]" = torch.ops.aten.reshape.default(mul_tensor_9, [-1]);  mul_tensor_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        var_mean_correction = torch.ops.aten.var_mean.correction(reshape_default, [0, 2], correction = 0, keepdim = True)
        getitem: "f32[1, 256, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 256, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_2: "f32[1, 256, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 256, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor: "f32[1, 256, 512]" = torch.ops.aten.sub.Tensor(reshape_default, getitem_1);  reshape_default = getitem_1 = None
        mul_tensor_10: "f32[1, 256, 512]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(reshape_default_1, -1);  reshape_default_1 = None
        mul_tensor_11: "f32[1, 256, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_10, unsqueeze_default);  mul_tensor_10 = unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        reshape_default_2: "f32[256, 512, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_11, _shape_param_1);  mul_tensor_11 = _shape_param_1 = None
        return (mul_tensor_8, reshape_default_2)

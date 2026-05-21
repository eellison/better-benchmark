class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[64, 3, 7, 7]", primals_2: "f32[64, 3, 32, 32]", primals_3: "f32[64]", primals_4: "f32[64]", primals_5: "f32[64, 64, 3, 3]", primals_6: "f32[64]", primals_7: "f32[64]", primals_8: "f32[64, 64, 3, 3]", primals_9: "f32[64]", primals_10: "f32[64]", primals_11: "f32[64, 64, 3, 3]", primals_12: "f32[64]", primals_13: "f32[64]", primals_14: "f32[64, 64, 3, 3]", primals_15: "f32[64]", primals_16: "f32[64]", primals_17: "f32[128, 64, 3, 3]", primals_18: "f32[128]", primals_19: "f32[128]", primals_20: "f32[128, 128, 3, 3]", primals_21: "f32[128]", primals_22: "f32[128]", primals_23: "f32[128, 64, 1, 1]", primals_24: "f32[128]", primals_25: "f32[128]", primals_26: "f32[128, 128, 3, 3]", primals_27: "f32[128]", primals_28: "f32[128]", primals_29: "f32[128, 128, 3, 3]", primals_30: "f32[128]", primals_31: "f32[128]", primals_32: "f32[256, 128, 3, 3]", primals_33: "f32[256]", primals_34: "f32[256]", primals_35: "f32[256, 256, 3, 3]", primals_36: "f32[256]", primals_37: "f32[256]", primals_38: "f32[256, 128, 1, 1]", primals_39: "f32[256]", primals_40: "f32[256]", primals_41: "f32[256, 256, 3, 3]", primals_42: "f32[256]", primals_43: "f32[256]", primals_44: "f32[256, 256, 3, 3]", primals_45: "f32[256]", primals_46: "f32[256]", primals_47: "f32[512, 256, 3, 3]", primals_48: "f32[512]", primals_49: "f32[512]", primals_50: "f32[512, 512, 3, 3]", primals_51: "f32[512]", primals_52: "f32[512]", primals_53: "f32[512, 256, 1, 1]", primals_54: "f32[512]", primals_55: "f32[512]", primals_56: "f32[512, 512, 3, 3]", primals_57: "f32[512]", primals_58: "f32[512]", primals_59: "f32[512, 512, 3, 3]", primals_60: "f32[512]", primals_61: "f32[512]", primals_62: "f32[1000, 512]", primals_63: "f32[1000]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:268 in _forward_impl, code: x = self.conv1(x)
        convolution: "f32[64, 64, 16, 16]" = torch.ops.aten.convolution.default(primals_2, primals_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:269 in _forward_impl, code: x = self.bn1(x)
        view: "f32[64, 32, 2, 256]" = torch.ops.aten.reshape.default(convolution, [64, 32, 2, 256])
        var_mean = torch.ops.aten.var_mean.correction(view, [2, 3], correction = 0, keepdim = True)
        getitem: "f32[64, 32, 1, 1]" = var_mean[0]
        getitem_1: "f32[64, 32, 1, 1]" = var_mean[1];  var_mean = None
        add: "f32[64, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[64, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[64, 32, 2, 256]" = torch.ops.aten.sub.Tensor(view, getitem_1);  view = None
        mul: "f32[64, 32, 2, 256]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        view_1: "f32[64, 64, 16, 16]" = torch.ops.aten.reshape.default(mul, [64, 64, 16, 16]);  mul = None
        unsqueeze: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_3, 0)
        unsqueeze_1: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        unsqueeze_2: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None
        mul_1: "f32[64, 64, 16, 16]" = torch.ops.aten.mul.Tensor(view_1, unsqueeze_2);  view_1 = unsqueeze_2 = None
        unsqueeze_3: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_4, 0)
        unsqueeze_4: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 2);  unsqueeze_3 = None
        unsqueeze_5: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3);  unsqueeze_4 = None
        add_1: "f32[64, 64, 16, 16]" = torch.ops.aten.add.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:270 in _forward_impl, code: x = self.relu(x)
        relu: "f32[64, 64, 16, 16]" = torch.ops.aten.relu.default(add_1);  add_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:271 in _forward_impl, code: x = self.maxpool(x)
        _low_memory_max_pool_with_offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu, [3, 3], [2, 2], [1, 1], [1, 1], False);  relu = None
        getitem_2: "f32[64, 64, 8, 8]" = _low_memory_max_pool_with_offsets[0]
        getitem_3: "i8[64, 64, 8, 8]" = _low_memory_max_pool_with_offsets[1];  _low_memory_max_pool_with_offsets = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:92 in forward, code: out = self.conv1(x)
        convolution_1: "f32[64, 64, 8, 8]" = torch.ops.aten.convolution.default(getitem_2, primals_5, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        view_2: "f32[64, 32, 2, 64]" = torch.ops.aten.reshape.default(convolution_1, [64, 32, 2, 64])
        var_mean_1 = torch.ops.aten.var_mean.correction(view_2, [2, 3], correction = 0, keepdim = True)
        getitem_4: "f32[64, 32, 1, 1]" = var_mean_1[0]
        getitem_5: "f32[64, 32, 1, 1]" = var_mean_1[1];  var_mean_1 = None
        add_2: "f32[64, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_1: "f32[64, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub_1: "f32[64, 32, 2, 64]" = torch.ops.aten.sub.Tensor(view_2, getitem_5);  view_2 = None
        mul_2: "f32[64, 32, 2, 64]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        view_3: "f32[64, 64, 8, 8]" = torch.ops.aten.reshape.default(mul_2, [64, 64, 8, 8]);  mul_2 = None
        unsqueeze_6: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_6, 0)
        unsqueeze_7: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 2);  unsqueeze_6 = None
        unsqueeze_8: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 3);  unsqueeze_7 = None
        mul_3: "f32[64, 64, 8, 8]" = torch.ops.aten.mul.Tensor(view_3, unsqueeze_8);  view_3 = unsqueeze_8 = None
        unsqueeze_9: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_7, 0);  primals_7 = None
        unsqueeze_10: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_9, 2);  unsqueeze_9 = None
        unsqueeze_11: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 3);  unsqueeze_10 = None
        add_3: "f32[64, 64, 8, 8]" = torch.ops.aten.add.Tensor(mul_3, unsqueeze_11);  mul_3 = unsqueeze_11 = None
        squeeze_2: "f32[64, 32]" = torch.ops.aten.squeeze.dims(getitem_5, [2, 3]);  getitem_5 = None
        squeeze_3: "f32[64, 32]" = torch.ops.aten.squeeze.dims(rsqrt_1, [2, 3]);  rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:94 in forward, code: out = self.relu(out)
        relu_1: "f32[64, 64, 8, 8]" = torch.ops.aten.relu.default(add_3);  add_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:96 in forward, code: out = self.conv2(out)
        convolution_2: "f32[64, 64, 8, 8]" = torch.ops.aten.convolution.default(relu_1, primals_8, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        view_4: "f32[64, 32, 2, 64]" = torch.ops.aten.reshape.default(convolution_2, [64, 32, 2, 64])
        var_mean_2 = torch.ops.aten.var_mean.correction(view_4, [2, 3], correction = 0, keepdim = True)
        getitem_6: "f32[64, 32, 1, 1]" = var_mean_2[0]
        getitem_7: "f32[64, 32, 1, 1]" = var_mean_2[1];  var_mean_2 = None
        add_4: "f32[64, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_2: "f32[64, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub_2: "f32[64, 32, 2, 64]" = torch.ops.aten.sub.Tensor(view_4, getitem_7);  view_4 = None
        mul_4: "f32[64, 32, 2, 64]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        view_5: "f32[64, 64, 8, 8]" = torch.ops.aten.reshape.default(mul_4, [64, 64, 8, 8]);  mul_4 = None
        unsqueeze_12: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_9, 0)
        unsqueeze_13: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, 2);  unsqueeze_12 = None
        unsqueeze_14: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_13, 3);  unsqueeze_13 = None
        mul_5: "f32[64, 64, 8, 8]" = torch.ops.aten.mul.Tensor(view_5, unsqueeze_14);  view_5 = unsqueeze_14 = None
        unsqueeze_15: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_10, 0);  primals_10 = None
        unsqueeze_16: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_15, 2);  unsqueeze_15 = None
        unsqueeze_17: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, 3);  unsqueeze_16 = None
        add_5: "f32[64, 64, 8, 8]" = torch.ops.aten.add.Tensor(mul_5, unsqueeze_17);  mul_5 = unsqueeze_17 = None
        squeeze_4: "f32[64, 32]" = torch.ops.aten.squeeze.dims(getitem_7, [2, 3]);  getitem_7 = None
        squeeze_5: "f32[64, 32]" = torch.ops.aten.squeeze.dims(rsqrt_2, [2, 3]);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:102 in forward, code: out += identity
        add_6: "f32[64, 64, 8, 8]" = torch.ops.aten.add.Tensor(add_5, getitem_2);  add_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        relu_2: "f32[64, 64, 8, 8]" = torch.ops.aten.relu.default(add_6);  add_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:92 in forward, code: out = self.conv1(x)
        convolution_3: "f32[64, 64, 8, 8]" = torch.ops.aten.convolution.default(relu_2, primals_11, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        view_6: "f32[64, 32, 2, 64]" = torch.ops.aten.reshape.default(convolution_3, [64, 32, 2, 64])
        var_mean_3 = torch.ops.aten.var_mean.correction(view_6, [2, 3], correction = 0, keepdim = True)
        getitem_8: "f32[64, 32, 1, 1]" = var_mean_3[0]
        getitem_9: "f32[64, 32, 1, 1]" = var_mean_3[1];  var_mean_3 = None
        add_7: "f32[64, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_3: "f32[64, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        sub_3: "f32[64, 32, 2, 64]" = torch.ops.aten.sub.Tensor(view_6, getitem_9);  view_6 = None
        mul_6: "f32[64, 32, 2, 64]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        view_7: "f32[64, 64, 8, 8]" = torch.ops.aten.reshape.default(mul_6, [64, 64, 8, 8]);  mul_6 = None
        unsqueeze_18: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_12, 0)
        unsqueeze_19: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, 2);  unsqueeze_18 = None
        unsqueeze_20: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_19, 3);  unsqueeze_19 = None
        mul_7: "f32[64, 64, 8, 8]" = torch.ops.aten.mul.Tensor(view_7, unsqueeze_20);  view_7 = unsqueeze_20 = None
        unsqueeze_21: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_13, 0);  primals_13 = None
        unsqueeze_22: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_21, 2);  unsqueeze_21 = None
        unsqueeze_23: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, 3);  unsqueeze_22 = None
        add_8: "f32[64, 64, 8, 8]" = torch.ops.aten.add.Tensor(mul_7, unsqueeze_23);  mul_7 = unsqueeze_23 = None
        squeeze_6: "f32[64, 32]" = torch.ops.aten.squeeze.dims(getitem_9, [2, 3]);  getitem_9 = None
        squeeze_7: "f32[64, 32]" = torch.ops.aten.squeeze.dims(rsqrt_3, [2, 3]);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:94 in forward, code: out = self.relu(out)
        relu_3: "f32[64, 64, 8, 8]" = torch.ops.aten.relu.default(add_8);  add_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:96 in forward, code: out = self.conv2(out)
        convolution_4: "f32[64, 64, 8, 8]" = torch.ops.aten.convolution.default(relu_3, primals_14, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        view_8: "f32[64, 32, 2, 64]" = torch.ops.aten.reshape.default(convolution_4, [64, 32, 2, 64])
        var_mean_4 = torch.ops.aten.var_mean.correction(view_8, [2, 3], correction = 0, keepdim = True)
        getitem_10: "f32[64, 32, 1, 1]" = var_mean_4[0]
        getitem_11: "f32[64, 32, 1, 1]" = var_mean_4[1];  var_mean_4 = None
        add_9: "f32[64, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
        rsqrt_4: "f32[64, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        sub_4: "f32[64, 32, 2, 64]" = torch.ops.aten.sub.Tensor(view_8, getitem_11);  view_8 = None
        mul_8: "f32[64, 32, 2, 64]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        view_9: "f32[64, 64, 8, 8]" = torch.ops.aten.reshape.default(mul_8, [64, 64, 8, 8]);  mul_8 = None
        unsqueeze_24: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_15, 0)
        unsqueeze_25: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, 2);  unsqueeze_24 = None
        unsqueeze_26: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_25, 3);  unsqueeze_25 = None
        mul_9: "f32[64, 64, 8, 8]" = torch.ops.aten.mul.Tensor(view_9, unsqueeze_26);  view_9 = unsqueeze_26 = None
        unsqueeze_27: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_16, 0);  primals_16 = None
        unsqueeze_28: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_27, 2);  unsqueeze_27 = None
        unsqueeze_29: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, 3);  unsqueeze_28 = None
        add_10: "f32[64, 64, 8, 8]" = torch.ops.aten.add.Tensor(mul_9, unsqueeze_29);  mul_9 = unsqueeze_29 = None
        squeeze_8: "f32[64, 32]" = torch.ops.aten.squeeze.dims(getitem_11, [2, 3]);  getitem_11 = None
        squeeze_9: "f32[64, 32]" = torch.ops.aten.squeeze.dims(rsqrt_4, [2, 3]);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:102 in forward, code: out += identity
        add_11: "f32[64, 64, 8, 8]" = torch.ops.aten.add.Tensor(add_10, relu_2);  add_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        relu_4: "f32[64, 64, 8, 8]" = torch.ops.aten.relu.default(add_11);  add_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:92 in forward, code: out = self.conv1(x)
        convolution_5: "f32[64, 128, 4, 4]" = torch.ops.aten.convolution.default(relu_4, primals_17, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        view_10: "f32[64, 32, 4, 16]" = torch.ops.aten.reshape.default(convolution_5, [64, 32, 4, 16])
        var_mean_5 = torch.ops.aten.var_mean.correction(view_10, [2, 3], correction = 0, keepdim = True)
        getitem_12: "f32[64, 32, 1, 1]" = var_mean_5[0]
        getitem_13: "f32[64, 32, 1, 1]" = var_mean_5[1];  var_mean_5 = None
        add_12: "f32[64, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
        rsqrt_5: "f32[64, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        sub_5: "f32[64, 32, 4, 16]" = torch.ops.aten.sub.Tensor(view_10, getitem_13);  view_10 = None
        mul_10: "f32[64, 32, 4, 16]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        view_11: "f32[64, 128, 4, 4]" = torch.ops.aten.reshape.default(mul_10, [64, 128, 4, 4]);  mul_10 = None
        unsqueeze_30: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_18, 0)
        unsqueeze_31: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, 2);  unsqueeze_30 = None
        unsqueeze_32: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_31, 3);  unsqueeze_31 = None
        mul_11: "f32[64, 128, 4, 4]" = torch.ops.aten.mul.Tensor(view_11, unsqueeze_32);  view_11 = unsqueeze_32 = None
        unsqueeze_33: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_19, 0);  primals_19 = None
        unsqueeze_34: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_33, 2);  unsqueeze_33 = None
        unsqueeze_35: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, 3);  unsqueeze_34 = None
        add_13: "f32[64, 128, 4, 4]" = torch.ops.aten.add.Tensor(mul_11, unsqueeze_35);  mul_11 = unsqueeze_35 = None
        squeeze_10: "f32[64, 32]" = torch.ops.aten.squeeze.dims(getitem_13, [2, 3]);  getitem_13 = None
        squeeze_11: "f32[64, 32]" = torch.ops.aten.squeeze.dims(rsqrt_5, [2, 3]);  rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:94 in forward, code: out = self.relu(out)
        relu_5: "f32[64, 128, 4, 4]" = torch.ops.aten.relu.default(add_13);  add_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:96 in forward, code: out = self.conv2(out)
        convolution_6: "f32[64, 128, 4, 4]" = torch.ops.aten.convolution.default(relu_5, primals_20, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        view_12: "f32[64, 32, 4, 16]" = torch.ops.aten.reshape.default(convolution_6, [64, 32, 4, 16])
        var_mean_6 = torch.ops.aten.var_mean.correction(view_12, [2, 3], correction = 0, keepdim = True)
        getitem_14: "f32[64, 32, 1, 1]" = var_mean_6[0]
        getitem_15: "f32[64, 32, 1, 1]" = var_mean_6[1];  var_mean_6 = None
        add_14: "f32[64, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_6: "f32[64, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        sub_6: "f32[64, 32, 4, 16]" = torch.ops.aten.sub.Tensor(view_12, getitem_15);  view_12 = None
        mul_12: "f32[64, 32, 4, 16]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
        view_13: "f32[64, 128, 4, 4]" = torch.ops.aten.reshape.default(mul_12, [64, 128, 4, 4]);  mul_12 = None
        unsqueeze_36: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_21, 0)
        unsqueeze_37: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, 2);  unsqueeze_36 = None
        unsqueeze_38: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_37, 3);  unsqueeze_37 = None
        mul_13: "f32[64, 128, 4, 4]" = torch.ops.aten.mul.Tensor(view_13, unsqueeze_38);  view_13 = unsqueeze_38 = None
        unsqueeze_39: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_22, 0);  primals_22 = None
        unsqueeze_40: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_39, 2);  unsqueeze_39 = None
        unsqueeze_41: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, 3);  unsqueeze_40 = None
        add_15: "f32[64, 128, 4, 4]" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_41);  mul_13 = unsqueeze_41 = None
        squeeze_12: "f32[64, 32]" = torch.ops.aten.squeeze.dims(getitem_15, [2, 3]);  getitem_15 = None
        squeeze_13: "f32[64, 32]" = torch.ops.aten.squeeze.dims(rsqrt_6, [2, 3]);  rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:100 in forward, code: identity = self.downsample(x)
        convolution_7: "f32[64, 128, 4, 4]" = torch.ops.aten.convolution.default(relu_4, primals_23, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        view_14: "f32[64, 32, 4, 16]" = torch.ops.aten.reshape.default(convolution_7, [64, 32, 4, 16])
        var_mean_7 = torch.ops.aten.var_mean.correction(view_14, [2, 3], correction = 0, keepdim = True)
        getitem_16: "f32[64, 32, 1, 1]" = var_mean_7[0]
        getitem_17: "f32[64, 32, 1, 1]" = var_mean_7[1];  var_mean_7 = None
        add_16: "f32[64, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_7: "f32[64, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        sub_7: "f32[64, 32, 4, 16]" = torch.ops.aten.sub.Tensor(view_14, getitem_17);  view_14 = None
        mul_14: "f32[64, 32, 4, 16]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        view_15: "f32[64, 128, 4, 4]" = torch.ops.aten.reshape.default(mul_14, [64, 128, 4, 4]);  mul_14 = None
        unsqueeze_42: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_24, 0)
        unsqueeze_43: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, 2);  unsqueeze_42 = None
        unsqueeze_44: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_43, 3);  unsqueeze_43 = None
        mul_15: "f32[64, 128, 4, 4]" = torch.ops.aten.mul.Tensor(view_15, unsqueeze_44);  view_15 = unsqueeze_44 = None
        unsqueeze_45: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_25, 0);  primals_25 = None
        unsqueeze_46: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_45, 2);  unsqueeze_45 = None
        unsqueeze_47: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, 3);  unsqueeze_46 = None
        add_17: "f32[64, 128, 4, 4]" = torch.ops.aten.add.Tensor(mul_15, unsqueeze_47);  mul_15 = unsqueeze_47 = None
        squeeze_14: "f32[64, 32]" = torch.ops.aten.squeeze.dims(getitem_17, [2, 3]);  getitem_17 = None
        squeeze_15: "f32[64, 32]" = torch.ops.aten.squeeze.dims(rsqrt_7, [2, 3]);  rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:102 in forward, code: out += identity
        add_18: "f32[64, 128, 4, 4]" = torch.ops.aten.add.Tensor(add_15, add_17);  add_15 = add_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        relu_6: "f32[64, 128, 4, 4]" = torch.ops.aten.relu.default(add_18);  add_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:92 in forward, code: out = self.conv1(x)
        convolution_8: "f32[64, 128, 4, 4]" = torch.ops.aten.convolution.default(relu_6, primals_26, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        view_16: "f32[64, 32, 4, 16]" = torch.ops.aten.reshape.default(convolution_8, [64, 32, 4, 16])
        var_mean_8 = torch.ops.aten.var_mean.correction(view_16, [2, 3], correction = 0, keepdim = True)
        getitem_18: "f32[64, 32, 1, 1]" = var_mean_8[0]
        getitem_19: "f32[64, 32, 1, 1]" = var_mean_8[1];  var_mean_8 = None
        add_19: "f32[64, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
        rsqrt_8: "f32[64, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
        sub_8: "f32[64, 32, 4, 16]" = torch.ops.aten.sub.Tensor(view_16, getitem_19);  view_16 = None
        mul_16: "f32[64, 32, 4, 16]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
        view_17: "f32[64, 128, 4, 4]" = torch.ops.aten.reshape.default(mul_16, [64, 128, 4, 4]);  mul_16 = None
        unsqueeze_48: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_27, 0)
        unsqueeze_49: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, 2);  unsqueeze_48 = None
        unsqueeze_50: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_49, 3);  unsqueeze_49 = None
        mul_17: "f32[64, 128, 4, 4]" = torch.ops.aten.mul.Tensor(view_17, unsqueeze_50);  view_17 = unsqueeze_50 = None
        unsqueeze_51: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_28, 0);  primals_28 = None
        unsqueeze_52: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_51, 2);  unsqueeze_51 = None
        unsqueeze_53: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, 3);  unsqueeze_52 = None
        add_20: "f32[64, 128, 4, 4]" = torch.ops.aten.add.Tensor(mul_17, unsqueeze_53);  mul_17 = unsqueeze_53 = None
        squeeze_16: "f32[64, 32]" = torch.ops.aten.squeeze.dims(getitem_19, [2, 3]);  getitem_19 = None
        squeeze_17: "f32[64, 32]" = torch.ops.aten.squeeze.dims(rsqrt_8, [2, 3]);  rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:94 in forward, code: out = self.relu(out)
        relu_7: "f32[64, 128, 4, 4]" = torch.ops.aten.relu.default(add_20);  add_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:96 in forward, code: out = self.conv2(out)
        convolution_9: "f32[64, 128, 4, 4]" = torch.ops.aten.convolution.default(relu_7, primals_29, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        view_18: "f32[64, 32, 4, 16]" = torch.ops.aten.reshape.default(convolution_9, [64, 32, 4, 16])
        var_mean_9 = torch.ops.aten.var_mean.correction(view_18, [2, 3], correction = 0, keepdim = True)
        getitem_20: "f32[64, 32, 1, 1]" = var_mean_9[0]
        getitem_21: "f32[64, 32, 1, 1]" = var_mean_9[1];  var_mean_9 = None
        add_21: "f32[64, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_9: "f32[64, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        sub_9: "f32[64, 32, 4, 16]" = torch.ops.aten.sub.Tensor(view_18, getitem_21);  view_18 = None
        mul_18: "f32[64, 32, 4, 16]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        view_19: "f32[64, 128, 4, 4]" = torch.ops.aten.reshape.default(mul_18, [64, 128, 4, 4]);  mul_18 = None
        unsqueeze_54: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_30, 0)
        unsqueeze_55: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, 2);  unsqueeze_54 = None
        unsqueeze_56: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_55, 3);  unsqueeze_55 = None
        mul_19: "f32[64, 128, 4, 4]" = torch.ops.aten.mul.Tensor(view_19, unsqueeze_56);  view_19 = unsqueeze_56 = None
        unsqueeze_57: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_31, 0);  primals_31 = None
        unsqueeze_58: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_57, 2);  unsqueeze_57 = None
        unsqueeze_59: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, 3);  unsqueeze_58 = None
        add_22: "f32[64, 128, 4, 4]" = torch.ops.aten.add.Tensor(mul_19, unsqueeze_59);  mul_19 = unsqueeze_59 = None
        squeeze_18: "f32[64, 32]" = torch.ops.aten.squeeze.dims(getitem_21, [2, 3]);  getitem_21 = None
        squeeze_19: "f32[64, 32]" = torch.ops.aten.squeeze.dims(rsqrt_9, [2, 3]);  rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:102 in forward, code: out += identity
        add_23: "f32[64, 128, 4, 4]" = torch.ops.aten.add.Tensor(add_22, relu_6);  add_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        relu_8: "f32[64, 128, 4, 4]" = torch.ops.aten.relu.default(add_23);  add_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:92 in forward, code: out = self.conv1(x)
        convolution_10: "f32[64, 256, 2, 2]" = torch.ops.aten.convolution.default(relu_8, primals_32, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        view_20: "f32[64, 32, 8, 4]" = torch.ops.aten.reshape.default(convolution_10, [64, 32, 8, 4])
        var_mean_10 = torch.ops.aten.var_mean.correction(view_20, [2, 3], correction = 0, keepdim = True)
        getitem_22: "f32[64, 32, 1, 1]" = var_mean_10[0]
        getitem_23: "f32[64, 32, 1, 1]" = var_mean_10[1];  var_mean_10 = None
        add_24: "f32[64, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_10: "f32[64, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        sub_10: "f32[64, 32, 8, 4]" = torch.ops.aten.sub.Tensor(view_20, getitem_23);  view_20 = None
        mul_20: "f32[64, 32, 8, 4]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        view_21: "f32[64, 256, 2, 2]" = torch.ops.aten.reshape.default(mul_20, [64, 256, 2, 2]);  mul_20 = None
        unsqueeze_60: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_33, 0)
        unsqueeze_61: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, 2);  unsqueeze_60 = None
        unsqueeze_62: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_61, 3);  unsqueeze_61 = None
        mul_21: "f32[64, 256, 2, 2]" = torch.ops.aten.mul.Tensor(view_21, unsqueeze_62);  view_21 = unsqueeze_62 = None
        unsqueeze_63: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_34, 0);  primals_34 = None
        unsqueeze_64: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_63, 2);  unsqueeze_63 = None
        unsqueeze_65: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, 3);  unsqueeze_64 = None
        add_25: "f32[64, 256, 2, 2]" = torch.ops.aten.add.Tensor(mul_21, unsqueeze_65);  mul_21 = unsqueeze_65 = None
        squeeze_20: "f32[64, 32]" = torch.ops.aten.squeeze.dims(getitem_23, [2, 3]);  getitem_23 = None
        squeeze_21: "f32[64, 32]" = torch.ops.aten.squeeze.dims(rsqrt_10, [2, 3]);  rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:94 in forward, code: out = self.relu(out)
        relu_9: "f32[64, 256, 2, 2]" = torch.ops.aten.relu.default(add_25);  add_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:96 in forward, code: out = self.conv2(out)
        convolution_11: "f32[64, 256, 2, 2]" = torch.ops.aten.convolution.default(relu_9, primals_35, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        view_22: "f32[64, 32, 8, 4]" = torch.ops.aten.reshape.default(convolution_11, [64, 32, 8, 4])
        var_mean_11 = torch.ops.aten.var_mean.correction(view_22, [2, 3], correction = 0, keepdim = True)
        getitem_24: "f32[64, 32, 1, 1]" = var_mean_11[0]
        getitem_25: "f32[64, 32, 1, 1]" = var_mean_11[1];  var_mean_11 = None
        add_26: "f32[64, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_11: "f32[64, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        sub_11: "f32[64, 32, 8, 4]" = torch.ops.aten.sub.Tensor(view_22, getitem_25);  view_22 = None
        mul_22: "f32[64, 32, 8, 4]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
        view_23: "f32[64, 256, 2, 2]" = torch.ops.aten.reshape.default(mul_22, [64, 256, 2, 2]);  mul_22 = None
        unsqueeze_66: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_36, 0)
        unsqueeze_67: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, 2);  unsqueeze_66 = None
        unsqueeze_68: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_67, 3);  unsqueeze_67 = None
        mul_23: "f32[64, 256, 2, 2]" = torch.ops.aten.mul.Tensor(view_23, unsqueeze_68);  view_23 = unsqueeze_68 = None
        unsqueeze_69: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_37, 0);  primals_37 = None
        unsqueeze_70: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_69, 2);  unsqueeze_69 = None
        unsqueeze_71: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, 3);  unsqueeze_70 = None
        add_27: "f32[64, 256, 2, 2]" = torch.ops.aten.add.Tensor(mul_23, unsqueeze_71);  mul_23 = unsqueeze_71 = None
        squeeze_22: "f32[64, 32]" = torch.ops.aten.squeeze.dims(getitem_25, [2, 3]);  getitem_25 = None
        squeeze_23: "f32[64, 32]" = torch.ops.aten.squeeze.dims(rsqrt_11, [2, 3]);  rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:100 in forward, code: identity = self.downsample(x)
        convolution_12: "f32[64, 256, 2, 2]" = torch.ops.aten.convolution.default(relu_8, primals_38, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        view_24: "f32[64, 32, 8, 4]" = torch.ops.aten.reshape.default(convolution_12, [64, 32, 8, 4])
        var_mean_12 = torch.ops.aten.var_mean.correction(view_24, [2, 3], correction = 0, keepdim = True)
        getitem_26: "f32[64, 32, 1, 1]" = var_mean_12[0]
        getitem_27: "f32[64, 32, 1, 1]" = var_mean_12[1];  var_mean_12 = None
        add_28: "f32[64, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_12: "f32[64, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        sub_12: "f32[64, 32, 8, 4]" = torch.ops.aten.sub.Tensor(view_24, getitem_27);  view_24 = None
        mul_24: "f32[64, 32, 8, 4]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
        view_25: "f32[64, 256, 2, 2]" = torch.ops.aten.reshape.default(mul_24, [64, 256, 2, 2]);  mul_24 = None
        unsqueeze_72: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_39, 0)
        unsqueeze_73: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, 2);  unsqueeze_72 = None
        unsqueeze_74: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_73, 3);  unsqueeze_73 = None
        mul_25: "f32[64, 256, 2, 2]" = torch.ops.aten.mul.Tensor(view_25, unsqueeze_74);  view_25 = unsqueeze_74 = None
        unsqueeze_75: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_40, 0);  primals_40 = None
        unsqueeze_76: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_75, 2);  unsqueeze_75 = None
        unsqueeze_77: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, 3);  unsqueeze_76 = None
        add_29: "f32[64, 256, 2, 2]" = torch.ops.aten.add.Tensor(mul_25, unsqueeze_77);  mul_25 = unsqueeze_77 = None
        squeeze_24: "f32[64, 32]" = torch.ops.aten.squeeze.dims(getitem_27, [2, 3]);  getitem_27 = None
        squeeze_25: "f32[64, 32]" = torch.ops.aten.squeeze.dims(rsqrt_12, [2, 3]);  rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:102 in forward, code: out += identity
        add_30: "f32[64, 256, 2, 2]" = torch.ops.aten.add.Tensor(add_27, add_29);  add_27 = add_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        relu_10: "f32[64, 256, 2, 2]" = torch.ops.aten.relu.default(add_30);  add_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:92 in forward, code: out = self.conv1(x)
        convolution_13: "f32[64, 256, 2, 2]" = torch.ops.aten.convolution.default(relu_10, primals_41, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        view_26: "f32[64, 32, 8, 4]" = torch.ops.aten.reshape.default(convolution_13, [64, 32, 8, 4])
        var_mean_13 = torch.ops.aten.var_mean.correction(view_26, [2, 3], correction = 0, keepdim = True)
        getitem_28: "f32[64, 32, 1, 1]" = var_mean_13[0]
        getitem_29: "f32[64, 32, 1, 1]" = var_mean_13[1];  var_mean_13 = None
        add_31: "f32[64, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
        rsqrt_13: "f32[64, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        sub_13: "f32[64, 32, 8, 4]" = torch.ops.aten.sub.Tensor(view_26, getitem_29);  view_26 = None
        mul_26: "f32[64, 32, 8, 4]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
        view_27: "f32[64, 256, 2, 2]" = torch.ops.aten.reshape.default(mul_26, [64, 256, 2, 2]);  mul_26 = None
        unsqueeze_78: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_42, 0)
        unsqueeze_79: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, 2);  unsqueeze_78 = None
        unsqueeze_80: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_79, 3);  unsqueeze_79 = None
        mul_27: "f32[64, 256, 2, 2]" = torch.ops.aten.mul.Tensor(view_27, unsqueeze_80);  view_27 = unsqueeze_80 = None
        unsqueeze_81: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_43, 0);  primals_43 = None
        unsqueeze_82: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_81, 2);  unsqueeze_81 = None
        unsqueeze_83: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, 3);  unsqueeze_82 = None
        add_32: "f32[64, 256, 2, 2]" = torch.ops.aten.add.Tensor(mul_27, unsqueeze_83);  mul_27 = unsqueeze_83 = None
        squeeze_26: "f32[64, 32]" = torch.ops.aten.squeeze.dims(getitem_29, [2, 3]);  getitem_29 = None
        squeeze_27: "f32[64, 32]" = torch.ops.aten.squeeze.dims(rsqrt_13, [2, 3]);  rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:94 in forward, code: out = self.relu(out)
        relu_11: "f32[64, 256, 2, 2]" = torch.ops.aten.relu.default(add_32);  add_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:96 in forward, code: out = self.conv2(out)
        convolution_14: "f32[64, 256, 2, 2]" = torch.ops.aten.convolution.default(relu_11, primals_44, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        view_28: "f32[64, 32, 8, 4]" = torch.ops.aten.reshape.default(convolution_14, [64, 32, 8, 4])
        var_mean_14 = torch.ops.aten.var_mean.correction(view_28, [2, 3], correction = 0, keepdim = True)
        getitem_30: "f32[64, 32, 1, 1]" = var_mean_14[0]
        getitem_31: "f32[64, 32, 1, 1]" = var_mean_14[1];  var_mean_14 = None
        add_33: "f32[64, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
        rsqrt_14: "f32[64, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        sub_14: "f32[64, 32, 8, 4]" = torch.ops.aten.sub.Tensor(view_28, getitem_31);  view_28 = None
        mul_28: "f32[64, 32, 8, 4]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
        view_29: "f32[64, 256, 2, 2]" = torch.ops.aten.reshape.default(mul_28, [64, 256, 2, 2]);  mul_28 = None
        unsqueeze_84: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_45, 0)
        unsqueeze_85: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, 2);  unsqueeze_84 = None
        unsqueeze_86: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_85, 3);  unsqueeze_85 = None
        mul_29: "f32[64, 256, 2, 2]" = torch.ops.aten.mul.Tensor(view_29, unsqueeze_86);  view_29 = unsqueeze_86 = None
        unsqueeze_87: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(primals_46, 0);  primals_46 = None
        unsqueeze_88: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_87, 2);  unsqueeze_87 = None
        unsqueeze_89: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, 3);  unsqueeze_88 = None
        add_34: "f32[64, 256, 2, 2]" = torch.ops.aten.add.Tensor(mul_29, unsqueeze_89);  mul_29 = unsqueeze_89 = None
        squeeze_28: "f32[64, 32]" = torch.ops.aten.squeeze.dims(getitem_31, [2, 3]);  getitem_31 = None
        squeeze_29: "f32[64, 32]" = torch.ops.aten.squeeze.dims(rsqrt_14, [2, 3]);  rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:102 in forward, code: out += identity
        add_35: "f32[64, 256, 2, 2]" = torch.ops.aten.add.Tensor(add_34, relu_10);  add_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        relu_12: "f32[64, 256, 2, 2]" = torch.ops.aten.relu.default(add_35);  add_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:92 in forward, code: out = self.conv1(x)
        convolution_15: "f32[64, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_12, primals_47, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        view_30: "f32[64, 32, 16, 1]" = torch.ops.aten.reshape.default(convolution_15, [64, 32, 16, 1])
        var_mean_15 = torch.ops.aten.var_mean.correction(view_30, [2, 3], correction = 0, keepdim = True)
        getitem_32: "f32[64, 32, 1, 1]" = var_mean_15[0]
        getitem_33: "f32[64, 32, 1, 1]" = var_mean_15[1];  var_mean_15 = None
        add_36: "f32[64, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_15: "f32[64, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_15: "f32[64, 32, 16, 1]" = torch.ops.aten.sub.Tensor(view_30, getitem_33);  view_30 = None
        mul_30: "f32[64, 32, 16, 1]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
        view_31: "f32[64, 512, 1, 1]" = torch.ops.aten.reshape.default(mul_30, [64, 512, 1, 1]);  mul_30 = None
        unsqueeze_90: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_48, 0)
        unsqueeze_91: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, 2);  unsqueeze_90 = None
        unsqueeze_92: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_91, 3);  unsqueeze_91 = None
        mul_31: "f32[64, 512, 1, 1]" = torch.ops.aten.mul.Tensor(view_31, unsqueeze_92);  view_31 = unsqueeze_92 = None
        unsqueeze_93: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_49, 0);  primals_49 = None
        unsqueeze_94: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_93, 2);  unsqueeze_93 = None
        unsqueeze_95: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, 3);  unsqueeze_94 = None
        add_37: "f32[64, 512, 1, 1]" = torch.ops.aten.add.Tensor(mul_31, unsqueeze_95);  mul_31 = unsqueeze_95 = None
        squeeze_30: "f32[64, 32]" = torch.ops.aten.squeeze.dims(getitem_33, [2, 3]);  getitem_33 = None
        squeeze_31: "f32[64, 32]" = torch.ops.aten.squeeze.dims(rsqrt_15, [2, 3]);  rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:94 in forward, code: out = self.relu(out)
        relu_13: "f32[64, 512, 1, 1]" = torch.ops.aten.relu.default(add_37);  add_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:96 in forward, code: out = self.conv2(out)
        convolution_16: "f32[64, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_13, primals_50, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        view_32: "f32[64, 32, 16, 1]" = torch.ops.aten.reshape.default(convolution_16, [64, 32, 16, 1])
        var_mean_16 = torch.ops.aten.var_mean.correction(view_32, [2, 3], correction = 0, keepdim = True)
        getitem_34: "f32[64, 32, 1, 1]" = var_mean_16[0]
        getitem_35: "f32[64, 32, 1, 1]" = var_mean_16[1];  var_mean_16 = None
        add_38: "f32[64, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
        rsqrt_16: "f32[64, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        sub_16: "f32[64, 32, 16, 1]" = torch.ops.aten.sub.Tensor(view_32, getitem_35);  view_32 = None
        mul_32: "f32[64, 32, 16, 1]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
        view_33: "f32[64, 512, 1, 1]" = torch.ops.aten.reshape.default(mul_32, [64, 512, 1, 1]);  mul_32 = None
        unsqueeze_96: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_51, 0)
        unsqueeze_97: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, 2);  unsqueeze_96 = None
        unsqueeze_98: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_97, 3);  unsqueeze_97 = None
        mul_33: "f32[64, 512, 1, 1]" = torch.ops.aten.mul.Tensor(view_33, unsqueeze_98);  view_33 = unsqueeze_98 = None
        unsqueeze_99: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_52, 0);  primals_52 = None
        unsqueeze_100: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_99, 2);  unsqueeze_99 = None
        unsqueeze_101: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, 3);  unsqueeze_100 = None
        add_39: "f32[64, 512, 1, 1]" = torch.ops.aten.add.Tensor(mul_33, unsqueeze_101);  mul_33 = unsqueeze_101 = None
        squeeze_32: "f32[64, 32]" = torch.ops.aten.squeeze.dims(getitem_35, [2, 3]);  getitem_35 = None
        squeeze_33: "f32[64, 32]" = torch.ops.aten.squeeze.dims(rsqrt_16, [2, 3]);  rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:100 in forward, code: identity = self.downsample(x)
        convolution_17: "f32[64, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_12, primals_53, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        view_34: "f32[64, 32, 16, 1]" = torch.ops.aten.reshape.default(convolution_17, [64, 32, 16, 1])
        var_mean_17 = torch.ops.aten.var_mean.correction(view_34, [2, 3], correction = 0, keepdim = True)
        getitem_36: "f32[64, 32, 1, 1]" = var_mean_17[0]
        getitem_37: "f32[64, 32, 1, 1]" = var_mean_17[1];  var_mean_17 = None
        add_40: "f32[64, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
        rsqrt_17: "f32[64, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        sub_17: "f32[64, 32, 16, 1]" = torch.ops.aten.sub.Tensor(view_34, getitem_37);  view_34 = None
        mul_34: "f32[64, 32, 16, 1]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
        view_35: "f32[64, 512, 1, 1]" = torch.ops.aten.reshape.default(mul_34, [64, 512, 1, 1]);  mul_34 = None
        unsqueeze_102: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_54, 0)
        unsqueeze_103: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, 2);  unsqueeze_102 = None
        unsqueeze_104: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_103, 3);  unsqueeze_103 = None
        mul_35: "f32[64, 512, 1, 1]" = torch.ops.aten.mul.Tensor(view_35, unsqueeze_104);  view_35 = unsqueeze_104 = None
        unsqueeze_105: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_55, 0);  primals_55 = None
        unsqueeze_106: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_105, 2);  unsqueeze_105 = None
        unsqueeze_107: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, 3);  unsqueeze_106 = None
        add_41: "f32[64, 512, 1, 1]" = torch.ops.aten.add.Tensor(mul_35, unsqueeze_107);  mul_35 = unsqueeze_107 = None
        squeeze_34: "f32[64, 32]" = torch.ops.aten.squeeze.dims(getitem_37, [2, 3]);  getitem_37 = None
        squeeze_35: "f32[64, 32]" = torch.ops.aten.squeeze.dims(rsqrt_17, [2, 3]);  rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:102 in forward, code: out += identity
        add_42: "f32[64, 512, 1, 1]" = torch.ops.aten.add.Tensor(add_39, add_41);  add_39 = add_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        relu_14: "f32[64, 512, 1, 1]" = torch.ops.aten.relu.default(add_42);  add_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:92 in forward, code: out = self.conv1(x)
        convolution_18: "f32[64, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_14, primals_56, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        view_36: "f32[64, 32, 16, 1]" = torch.ops.aten.reshape.default(convolution_18, [64, 32, 16, 1])
        var_mean_18 = torch.ops.aten.var_mean.correction(view_36, [2, 3], correction = 0, keepdim = True)
        getitem_38: "f32[64, 32, 1, 1]" = var_mean_18[0]
        getitem_39: "f32[64, 32, 1, 1]" = var_mean_18[1];  var_mean_18 = None
        add_43: "f32[64, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
        rsqrt_18: "f32[64, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        sub_18: "f32[64, 32, 16, 1]" = torch.ops.aten.sub.Tensor(view_36, getitem_39);  view_36 = None
        mul_36: "f32[64, 32, 16, 1]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
        view_37: "f32[64, 512, 1, 1]" = torch.ops.aten.reshape.default(mul_36, [64, 512, 1, 1]);  mul_36 = None
        unsqueeze_108: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_57, 0)
        unsqueeze_109: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, 2);  unsqueeze_108 = None
        unsqueeze_110: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_109, 3);  unsqueeze_109 = None
        mul_37: "f32[64, 512, 1, 1]" = torch.ops.aten.mul.Tensor(view_37, unsqueeze_110);  view_37 = unsqueeze_110 = None
        unsqueeze_111: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_58, 0);  primals_58 = None
        unsqueeze_112: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_111, 2);  unsqueeze_111 = None
        unsqueeze_113: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, 3);  unsqueeze_112 = None
        add_44: "f32[64, 512, 1, 1]" = torch.ops.aten.add.Tensor(mul_37, unsqueeze_113);  mul_37 = unsqueeze_113 = None
        squeeze_36: "f32[64, 32]" = torch.ops.aten.squeeze.dims(getitem_39, [2, 3]);  getitem_39 = None
        squeeze_37: "f32[64, 32]" = torch.ops.aten.squeeze.dims(rsqrt_18, [2, 3]);  rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:94 in forward, code: out = self.relu(out)
        relu_15: "f32[64, 512, 1, 1]" = torch.ops.aten.relu.default(add_44);  add_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:96 in forward, code: out = self.conv2(out)
        convolution_19: "f32[64, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_15, primals_59, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        view_38: "f32[64, 32, 16, 1]" = torch.ops.aten.reshape.default(convolution_19, [64, 32, 16, 1])
        var_mean_19 = torch.ops.aten.var_mean.correction(view_38, [2, 3], correction = 0, keepdim = True)
        getitem_40: "f32[64, 32, 1, 1]" = var_mean_19[0]
        getitem_41: "f32[64, 32, 1, 1]" = var_mean_19[1];  var_mean_19 = None
        add_45: "f32[64, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
        rsqrt_19: "f32[64, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        sub_19: "f32[64, 32, 16, 1]" = torch.ops.aten.sub.Tensor(view_38, getitem_41);  view_38 = None
        mul_38: "f32[64, 32, 16, 1]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
        view_39: "f32[64, 512, 1, 1]" = torch.ops.aten.reshape.default(mul_38, [64, 512, 1, 1]);  mul_38 = None
        unsqueeze_114: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_60, 0)
        unsqueeze_115: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, 2);  unsqueeze_114 = None
        unsqueeze_116: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_115, 3);  unsqueeze_115 = None
        mul_39: "f32[64, 512, 1, 1]" = torch.ops.aten.mul.Tensor(view_39, unsqueeze_116);  view_39 = unsqueeze_116 = None
        unsqueeze_117: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_61, 0);  primals_61 = None
        unsqueeze_118: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_117, 2);  unsqueeze_117 = None
        unsqueeze_119: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, 3);  unsqueeze_118 = None
        add_46: "f32[64, 512, 1, 1]" = torch.ops.aten.add.Tensor(mul_39, unsqueeze_119);  mul_39 = unsqueeze_119 = None
        squeeze_38: "f32[64, 32]" = torch.ops.aten.squeeze.dims(getitem_41, [2, 3]);  getitem_41 = None
        squeeze_39: "f32[64, 32]" = torch.ops.aten.squeeze.dims(rsqrt_19, [2, 3]);  rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:102 in forward, code: out += identity
        add_47: "f32[64, 512, 1, 1]" = torch.ops.aten.add.Tensor(add_46, relu_14);  add_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        relu_16: "f32[64, 512, 1, 1]" = torch.ops.aten.relu.default(add_47);  add_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:278 in _forward_impl, code: x = self.avgpool(x)
        mean: "f32[64, 512, 1, 1]" = torch.ops.aten.mean.dim(relu_16, [-1, -2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:279 in _forward_impl, code: x = torch.flatten(x, 1)
        view_40: "f32[64, 512]" = torch.ops.aten.reshape.default(mean, [64, 512]);  mean = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:280 in _forward_impl, code: x = self.fc(x)
        permute: "f32[512, 1000]" = torch.ops.aten.permute.default(primals_62, [1, 0])
        addmm: "f32[64, 1000]" = torch.ops.aten.addmm.default(primals_63, view_40, permute);  primals_63 = permute = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        le: "b8[64, 512, 1, 1]" = torch.ops.aten.le.Scalar(relu_16, 0);  relu_16 = None
        return (addmm, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_8, primals_9, primals_11, primals_12, primals_14, primals_15, primals_17, primals_18, primals_20, primals_21, primals_23, primals_24, primals_26, primals_27, primals_29, primals_30, primals_32, primals_33, primals_35, primals_36, primals_38, primals_39, primals_41, primals_42, primals_44, primals_45, primals_47, primals_48, primals_50, primals_51, primals_53, primals_54, primals_56, primals_57, primals_59, primals_60, primals_62, convolution, getitem_1, rsqrt, getitem_2, getitem_3, convolution_1, squeeze_2, squeeze_3, relu_1, convolution_2, squeeze_4, squeeze_5, relu_2, convolution_3, squeeze_6, squeeze_7, relu_3, convolution_4, squeeze_8, squeeze_9, relu_4, convolution_5, squeeze_10, squeeze_11, relu_5, convolution_6, squeeze_12, squeeze_13, convolution_7, squeeze_14, squeeze_15, relu_6, convolution_8, squeeze_16, squeeze_17, relu_7, convolution_9, squeeze_18, squeeze_19, relu_8, convolution_10, squeeze_20, squeeze_21, relu_9, convolution_11, squeeze_22, squeeze_23, convolution_12, squeeze_24, squeeze_25, relu_10, convolution_13, squeeze_26, squeeze_27, relu_11, convolution_14, squeeze_28, squeeze_29, relu_12, convolution_15, squeeze_30, squeeze_31, relu_13, convolution_16, squeeze_32, squeeze_33, convolution_17, squeeze_34, squeeze_35, relu_14, convolution_18, squeeze_36, squeeze_37, relu_15, convolution_19, squeeze_38, squeeze_39, view_40, le)

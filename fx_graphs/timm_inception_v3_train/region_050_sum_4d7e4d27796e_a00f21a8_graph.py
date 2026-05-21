class GraphModule(torch.nn.Module):
    def forward(self, add_487: "f32[128, 768, 17, 17]", convolution_49: "f32[128, 192, 17, 17]", getitem_105: "f32[1, 192, 1, 1]", rsqrt_49: "f32[1, 192, 1, 1]", primals_300: "f32[192]", primals_301: "f32[192]", full_default: "f32[]", relu_44: "f32[128, 160, 17, 17]", getitem_340: "f32[128, 160, 17, 17]", convolution_44: "f32[128, 160, 17, 17]", unsqueeze_966: "f32[1, 160, 1, 1]", squeeze_133: "f32[160]", primals_270: "f32[160]", relu_41: "f32[128, 160, 17, 17]", getitem_349: "f32[128, 160, 17, 17]", convolution_41: "f32[128, 160, 17, 17]", unsqueeze_1002: "f32[1, 160, 1, 1]", squeeze_124: "f32[160]", primals_252: "f32[160]", convolution_40: "f32[128, 192, 17, 17]", getitem_87: "f32[1, 192, 1, 1]", rsqrt_40: "f32[1, 192, 1, 1]", primals_246: "f32[192]", primals_247: "f32[192]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:152 in forward, code: return torch.cat(outputs, 1)
        slice_tensor: "f32[128, 192, 17, 17]" = torch.ops.aten.slice.Tensor(add_487, 1, 0, 192)
        slice_tensor_1: "f32[128, 192, 17, 17]" = torch.ops.aten.slice.Tensor(add_487, 1, 576, 768);  add_487 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_tensor: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_49, getitem_105)
        mul_tensor: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_49);  sub_tensor = None
        unsqueeze_default: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_300, -1)
        unsqueeze_default_1: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_301, -1);  primals_301 = None
        unsqueeze_default_3: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor: "f32[128, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default: "f32[128, 192, 17, 17]" = torch.ops.aten.relu.default(add_tensor);  add_tensor = None
        le_scalar: "b8[128, 192, 17, 17]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        where_self: "f32[128, 192, 17, 17]" = torch.ops.aten.where.self(le_scalar, full_default, slice_tensor_1);  le_scalar = slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_105, [0, 2, 3]);  getitem_105 = None
        unsqueeze_default_4: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_5: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        sum_dim_int_list: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_1: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_default_6);  convolution_49 = unsqueeze_default_6 = None
        mul_tensor_2: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_1)
        sum_dim_int_list_1: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 2, 3]);  mul_tensor_2 = None
        mul_tensor_3: "f32[192]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 2.703287197231834e-05);  sum_dim_int_list = None
        unsqueeze_default_7: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, 0);  mul_tensor_3 = None
        unsqueeze_default_8: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        mul_tensor_4: "f32[192]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 2.703287197231834e-05);  sum_dim_int_list_1 = None
        squeeze_dims_1: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_49, [0, 2, 3]);  rsqrt_49 = None
        mul_tensor_5: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_6: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default_10: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, 0);  mul_tensor_6 = None
        unsqueeze_default_11: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        mul_tensor_7: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, primals_300);  squeeze_dims_1 = primals_300 = None
        unsqueeze_default_13: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_7, 0);  mul_tensor_7 = None
        unsqueeze_default_14: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        mul_tensor_8: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_12);  sub_tensor_1 = unsqueeze_default_12 = None
        sub_tensor_2: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_8);  where_self = mul_tensor_8 = None
        sub_tensor_3: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_tensor_2, unsqueeze_default_9);  sub_tensor_2 = unsqueeze_default_9 = None
        mul_tensor_9: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_15);  sub_tensor_3 = unsqueeze_default_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_scalar_1: "b8[128, 160, 17, 17]" = torch.ops.aten.le.Scalar(relu_44, 0);  relu_44 = None
        where_self_1: "f32[128, 160, 17, 17]" = torch.ops.aten.where.self(le_scalar_1, full_default, getitem_340);  le_scalar_1 = getitem_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_dim_int_list_2: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_self_1, [0, 2, 3])
        sub_tensor_4: "f32[128, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_966);  convolution_44 = unsqueeze_966 = None
        mul_tensor_10: "f32[128, 160, 17, 17]" = torch.ops.aten.mul.Tensor(where_self_1, sub_tensor_4)
        sum_dim_int_list_3: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 2, 3]);  mul_tensor_10 = None
        mul_tensor_11: "f32[160]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_2, 2.703287197231834e-05);  sum_dim_int_list_2 = None
        unsqueeze_default_16: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_tensor_11, 0);  mul_tensor_11 = None
        unsqueeze_default_17: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, 2);  unsqueeze_default_16 = None
        unsqueeze_default_18: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_17, 3);  unsqueeze_default_17 = None
        mul_tensor_12: "f32[160]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, 2.703287197231834e-05);  sum_dim_int_list_3 = None
        mul_tensor_13: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
        mul_tensor_14: "f32[160]" = torch.ops.aten.mul.Tensor(mul_tensor_12, mul_tensor_13);  mul_tensor_12 = mul_tensor_13 = None
        unsqueeze_default_19: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_tensor_14, 0);  mul_tensor_14 = None
        unsqueeze_default_20: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_19, 2);  unsqueeze_default_19 = None
        unsqueeze_default_21: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_20, 3);  unsqueeze_default_20 = None
        mul_tensor_15: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_133, primals_270);  squeeze_133 = primals_270 = None
        unsqueeze_default_22: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_tensor_15, 0);  mul_tensor_15 = None
        unsqueeze_default_23: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_22, 2);  unsqueeze_default_22 = None
        unsqueeze_default_24: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_23, 3);  unsqueeze_default_23 = None
        mul_tensor_16: "f32[128, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_4, unsqueeze_default_21);  sub_tensor_4 = unsqueeze_default_21 = None
        sub_tensor_5: "f32[128, 160, 17, 17]" = torch.ops.aten.sub.Tensor(where_self_1, mul_tensor_16);  where_self_1 = mul_tensor_16 = None
        sub_tensor_6: "f32[128, 160, 17, 17]" = torch.ops.aten.sub.Tensor(sub_tensor_5, unsqueeze_default_18);  sub_tensor_5 = unsqueeze_default_18 = None
        mul_tensor_17: "f32[128, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_6, unsqueeze_default_24);  sub_tensor_6 = unsqueeze_default_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_scalar_2: "b8[128, 160, 17, 17]" = torch.ops.aten.le.Scalar(relu_41, 0);  relu_41 = None
        where_self_2: "f32[128, 160, 17, 17]" = torch.ops.aten.where.self(le_scalar_2, full_default, getitem_349);  le_scalar_2 = getitem_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_dim_int_list_4: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_self_2, [0, 2, 3])
        sub_tensor_7: "f32[128, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_1002);  convolution_41 = unsqueeze_1002 = None
        mul_tensor_18: "f32[128, 160, 17, 17]" = torch.ops.aten.mul.Tensor(where_self_2, sub_tensor_7)
        sum_dim_int_list_5: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_tensor_18, [0, 2, 3]);  mul_tensor_18 = None
        mul_tensor_19: "f32[160]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_4, 2.703287197231834e-05);  sum_dim_int_list_4 = None
        unsqueeze_default_25: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_tensor_19, 0);  mul_tensor_19 = None
        unsqueeze_default_26: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_25, 2);  unsqueeze_default_25 = None
        unsqueeze_default_27: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_26, 3);  unsqueeze_default_26 = None
        mul_tensor_20: "f32[160]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_5, 2.703287197231834e-05);  sum_dim_int_list_5 = None
        mul_tensor_21: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
        mul_tensor_22: "f32[160]" = torch.ops.aten.mul.Tensor(mul_tensor_20, mul_tensor_21);  mul_tensor_20 = mul_tensor_21 = None
        unsqueeze_default_28: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_tensor_22, 0);  mul_tensor_22 = None
        unsqueeze_default_29: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_28, 2);  unsqueeze_default_28 = None
        unsqueeze_default_30: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_29, 3);  unsqueeze_default_29 = None
        mul_tensor_23: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_124, primals_252);  squeeze_124 = primals_252 = None
        unsqueeze_default_31: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_tensor_23, 0);  mul_tensor_23 = None
        unsqueeze_default_32: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_31, 2);  unsqueeze_default_31 = None
        unsqueeze_default_33: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_32, 3);  unsqueeze_default_32 = None
        mul_tensor_24: "f32[128, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_7, unsqueeze_default_30);  sub_tensor_7 = unsqueeze_default_30 = None
        sub_tensor_8: "f32[128, 160, 17, 17]" = torch.ops.aten.sub.Tensor(where_self_2, mul_tensor_24);  where_self_2 = mul_tensor_24 = None
        sub_tensor_9: "f32[128, 160, 17, 17]" = torch.ops.aten.sub.Tensor(sub_tensor_8, unsqueeze_default_27);  sub_tensor_8 = unsqueeze_default_27 = None
        mul_tensor_25: "f32[128, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_9, unsqueeze_default_33);  sub_tensor_9 = unsqueeze_default_33 = None
        sub_tensor_10: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_40, getitem_87)
        mul_tensor_26: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_10, rsqrt_40);  sub_tensor_10 = None
        unsqueeze_default_34: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_246, -1)
        unsqueeze_default_35: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_34, -1);  unsqueeze_default_34 = None
        mul_tensor_27: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_tensor_26, unsqueeze_default_35);  mul_tensor_26 = unsqueeze_default_35 = None
        unsqueeze_default_36: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_247, -1);  primals_247 = None
        unsqueeze_default_37: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_36, -1);  unsqueeze_default_36 = None
        add_tensor_1: "f32[128, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_tensor_27, unsqueeze_default_37);  mul_tensor_27 = unsqueeze_default_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default_1: "f32[128, 192, 17, 17]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None
        le_scalar_3: "b8[128, 192, 17, 17]" = torch.ops.aten.le.Scalar(relu_default_1, 0);  relu_default_1 = None
        where_self_3: "f32[128, 192, 17, 17]" = torch.ops.aten.where.self(le_scalar_3, full_default, slice_tensor);  le_scalar_3 = full_default = slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_2: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
        unsqueeze_default_38: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims_2, 0);  squeeze_dims_2 = None
        unsqueeze_default_39: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_38, 2);  unsqueeze_default_38 = None
        unsqueeze_default_40: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_39, 3);  unsqueeze_default_39 = None
        sum_dim_int_list_6: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_self_3, [0, 2, 3])
        sub_tensor_11: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_default_40);  convolution_40 = unsqueeze_default_40 = None
        mul_tensor_28: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_self_3, sub_tensor_11)
        sum_dim_int_list_7: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_28, [0, 2, 3]);  mul_tensor_28 = None
        mul_tensor_29: "f32[192]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_6, 2.703287197231834e-05);  sum_dim_int_list_6 = None
        unsqueeze_default_41: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_29, 0);  mul_tensor_29 = None
        unsqueeze_default_42: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_41, 2);  unsqueeze_default_41 = None
        unsqueeze_default_43: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_42, 3);  unsqueeze_default_42 = None
        mul_tensor_30: "f32[192]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_7, 2.703287197231834e-05);  sum_dim_int_list_7 = None
        squeeze_dims_3: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
        mul_tensor_31: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, squeeze_dims_3)
        mul_tensor_32: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_30, mul_tensor_31);  mul_tensor_30 = mul_tensor_31 = None
        unsqueeze_default_44: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_32, 0);  mul_tensor_32 = None
        unsqueeze_default_45: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_44, 2);  unsqueeze_default_44 = None
        unsqueeze_default_46: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_45, 3);  unsqueeze_default_45 = None
        mul_tensor_33: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, primals_246);  squeeze_dims_3 = primals_246 = None
        unsqueeze_default_47: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_33, 0);  mul_tensor_33 = None
        unsqueeze_default_48: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_47, 2);  unsqueeze_default_47 = None
        unsqueeze_default_49: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_48, 3);  unsqueeze_default_48 = None
        mul_tensor_34: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_11, unsqueeze_default_46);  sub_tensor_11 = unsqueeze_default_46 = None
        sub_tensor_12: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_self_3, mul_tensor_34);  where_self_3 = mul_tensor_34 = None
        sub_tensor_13: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_tensor_12, unsqueeze_default_43);  sub_tensor_12 = unsqueeze_default_43 = None
        mul_tensor_35: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_13, unsqueeze_default_49);  sub_tensor_13 = unsqueeze_default_49 = None
        return (mul_tensor_9, mul_tensor_17, mul_tensor_25, mul_tensor_35)

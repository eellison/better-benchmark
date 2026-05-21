class GraphModule(torch.nn.Module):
    def forward(self, mm: "f32[128, 1408]", convolution_42: "f32[128, 1408, 7, 7]", getitem_119: "f32[1, 1408, 1, 1]", rsqrt_59: "f32[1, 1408, 1, 1]", primals_343: "f32[1408]", primals_344: "f32[1408]", convolution_43: "f32[128, 1408, 7, 7]", getitem_121: "f32[1, 1408, 1, 1]", rsqrt_60: "f32[1, 1408, 1, 1]", primals_349: "f32[1408]", primals_350: "f32[1408]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        reshape_default: "f32[128, 1408, 1, 1]" = torch.ops.aten.reshape.default(mm, _shape_param_0);  mm = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        squeeze_dim: "f32[128, 1408, 1]" = torch.ops.aten.squeeze.dim(reshape_default, 3);  reshape_default = None
        squeeze_dim_1: "f32[128, 1408]" = torch.ops.aten.squeeze.dim(squeeze_dim, 2);  squeeze_dim = None
        full_default: "f32[180224]" = torch.ops.aten.full.default([180224], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        as_strided_scatter_default: "f32[180224]" = torch.ops.aten.as_strided_scatter.default(full_default, squeeze_dim_1, [128, 1408], [1408, 1], 0);  full_default = squeeze_dim_1 = None
        as_strided_default: "f32[128, 1408, 1, 1]" = torch.ops.aten.as_strided.default(as_strided_scatter_default, [128, 1408, 1, 1], [1408, 1, 1, 1], 0);  as_strided_scatter_default = None
        expand_default: "f32[128, 1408, 7, 7]" = torch.ops.aten.expand.default(as_strided_default, _shape_param_1);  as_strided_default = _shape_param_1 = None
        div_scalar: "f32[128, 1408, 7, 7]" = torch.ops.aten.div.Scalar(expand_default, 49);  expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_tensor: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_42, getitem_119)
        mul_tensor: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_59);  sub_tensor = None
        unsqueeze_default: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(primals_343, -1)
        unsqueeze_default_1: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(primals_344, -1);  primals_344 = None
        unsqueeze_default_3: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor: "f32[128, 1408, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        sub_tensor_1: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_43, getitem_121)
        mul_tensor_2: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_60);  sub_tensor_1 = None
        unsqueeze_default_4: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(primals_349, -1)
        unsqueeze_default_5: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_3: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_2, unsqueeze_default_5);  mul_tensor_2 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(primals_350, -1);  primals_350 = None
        unsqueeze_default_7: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[128, 1408, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_3, unsqueeze_default_7);  mul_tensor_3 = unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:748 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_tensor_2: "f32[128, 1408, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor, add_tensor_1);  add_tensor = add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        relu_default: "f32[128, 1408, 7, 7]" = torch.ops.aten.relu.default(add_tensor_2);  add_tensor_2 = None
        le_scalar: "b8[128, 1408, 7, 7]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[128, 1408, 7, 7]" = torch.ops.aten.where.self(le_scalar, full_default_1, div_scalar);  le_scalar = full_default_1 = div_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims: "f32[1408]" = torch.ops.aten.squeeze.dims(getitem_121, [0, 2, 3]);  getitem_121 = None
        unsqueeze_default_8: "f32[1, 1408]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_9: "f32[1, 1408, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 2);  unsqueeze_default_8 = None
        unsqueeze_default_10: "f32[1, 1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, 3);  unsqueeze_default_9 = None
        sum_dim_int_list: "f32[1408]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_2: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_default_10);  convolution_43 = unsqueeze_default_10 = None
        mul_tensor_4: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_2)
        sum_dim_int_list_1: "f32[1408]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 2, 3]);  mul_tensor_4 = None
        mul_tensor_5: "f32[1408]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.00015943877551020407);  sum_dim_int_list = None
        unsqueeze_default_11: "f32[1, 1408]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_12: "f32[1, 1408, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 2);  unsqueeze_default_11 = None
        unsqueeze_default_13: "f32[1, 1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, 3);  unsqueeze_default_12 = None
        mul_tensor_6: "f32[1408]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 0.00015943877551020407);  sum_dim_int_list_1 = None
        squeeze_dims_1: "f32[1408]" = torch.ops.aten.squeeze.dims(rsqrt_60, [0, 2, 3]);  rsqrt_60 = None
        mul_tensor_7: "f32[1408]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_8: "f32[1408]" = torch.ops.aten.mul.Tensor(mul_tensor_6, mul_tensor_7);  mul_tensor_6 = mul_tensor_7 = None
        unsqueeze_default_14: "f32[1, 1408]" = torch.ops.aten.unsqueeze.default(mul_tensor_8, 0);  mul_tensor_8 = None
        unsqueeze_default_15: "f32[1, 1408, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 2);  unsqueeze_default_14 = None
        unsqueeze_default_16: "f32[1, 1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_15, 3);  unsqueeze_default_15 = None
        mul_tensor_9: "f32[1408]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, primals_349);  squeeze_dims_1 = primals_349 = None
        unsqueeze_default_17: "f32[1, 1408]" = torch.ops.aten.unsqueeze.default(mul_tensor_9, 0);  mul_tensor_9 = None
        unsqueeze_default_18: "f32[1, 1408, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_17, 2);  unsqueeze_default_17 = None
        unsqueeze_default_19: "f32[1, 1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_18, 3);  unsqueeze_default_18 = None
        mul_tensor_10: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_16);  sub_tensor_2 = unsqueeze_default_16 = None
        sub_tensor_3: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_10);  mul_tensor_10 = None
        sub_tensor_4: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(sub_tensor_3, unsqueeze_default_13);  sub_tensor_3 = unsqueeze_default_13 = None
        mul_tensor_11: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_4, unsqueeze_default_19);  sub_tensor_4 = unsqueeze_default_19 = None
        squeeze_dims_2: "f32[1408]" = torch.ops.aten.squeeze.dims(getitem_119, [0, 2, 3]);  getitem_119 = None
        unsqueeze_default_20: "f32[1, 1408]" = torch.ops.aten.unsqueeze.default(squeeze_dims_2, 0);  squeeze_dims_2 = None
        unsqueeze_default_21: "f32[1, 1408, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_20, 2);  unsqueeze_default_20 = None
        unsqueeze_default_22: "f32[1, 1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_21, 3);  unsqueeze_default_21 = None
        sum_dim_int_list_2: "f32[1408]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_5: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_default_22);  convolution_42 = unsqueeze_default_22 = None
        mul_tensor_12: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_5)
        sum_dim_int_list_3: "f32[1408]" = torch.ops.aten.sum.dim_IntList(mul_tensor_12, [0, 2, 3]);  mul_tensor_12 = None
        mul_tensor_13: "f32[1408]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_2, 0.00015943877551020407);  sum_dim_int_list_2 = None
        unsqueeze_default_23: "f32[1, 1408]" = torch.ops.aten.unsqueeze.default(mul_tensor_13, 0);  mul_tensor_13 = None
        unsqueeze_default_24: "f32[1, 1408, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_23, 2);  unsqueeze_default_23 = None
        unsqueeze_default_25: "f32[1, 1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_24, 3);  unsqueeze_default_24 = None
        mul_tensor_14: "f32[1408]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, 0.00015943877551020407);  sum_dim_int_list_3 = None
        squeeze_dims_3: "f32[1408]" = torch.ops.aten.squeeze.dims(rsqrt_59, [0, 2, 3]);  rsqrt_59 = None
        mul_tensor_15: "f32[1408]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, squeeze_dims_3)
        mul_tensor_16: "f32[1408]" = torch.ops.aten.mul.Tensor(mul_tensor_14, mul_tensor_15);  mul_tensor_14 = mul_tensor_15 = None
        unsqueeze_default_26: "f32[1, 1408]" = torch.ops.aten.unsqueeze.default(mul_tensor_16, 0);  mul_tensor_16 = None
        unsqueeze_default_27: "f32[1, 1408, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_26, 2);  unsqueeze_default_26 = None
        unsqueeze_default_28: "f32[1, 1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_27, 3);  unsqueeze_default_27 = None
        mul_tensor_17: "f32[1408]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, primals_343);  squeeze_dims_3 = primals_343 = None
        unsqueeze_default_29: "f32[1, 1408]" = torch.ops.aten.unsqueeze.default(mul_tensor_17, 0);  mul_tensor_17 = None
        unsqueeze_default_30: "f32[1, 1408, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_29, 2);  unsqueeze_default_29 = None
        unsqueeze_default_31: "f32[1, 1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_30, 3);  unsqueeze_default_30 = None
        mul_tensor_18: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_5, unsqueeze_default_28);  sub_tensor_5 = unsqueeze_default_28 = None
        sub_tensor_6: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_18);  where_self = mul_tensor_18 = None
        sub_tensor_7: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(sub_tensor_6, unsqueeze_default_25);  sub_tensor_6 = unsqueeze_default_25 = None
        mul_tensor_19: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_7, unsqueeze_default_31);  sub_tensor_7 = unsqueeze_default_31 = None
        return (mul_tensor_11, mul_tensor_19)

class GraphModule(torch.nn.Module):
    def forward(self, addmm_45: "f32[8192, 1024]", add_248: "f32[512, 16, 1024]", arg345_1: "f32[1024]", arg346_1: "f32[1024]", arg347_1: "f32[1024, 16, 64]", arg348_1: "f32[1024, 16, 64]", clone_2: "f32[1024, 16, 1024]", arg350_1: "f32[1024, 16, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        reshape_default: "f32[512, 16, 1024]" = torch.ops.aten.reshape.default(addmm_45, _shape_param_0);  addmm_45 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_tensor: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(reshape_default, add_248);  reshape_default = add_248 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[512, 16, 1]" = var_mean_correction[0]
        getitem_1: "f32[512, 16, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        add_tensor_1: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, arg345_1);  mul_tensor = arg345_1 = None
        add_tensor_2: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg346_1);  mul_tensor_1 = arg346_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        unsqueeze_default: "f32[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_tensor_2, 3)
        unsqueeze_default_1: "f32[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 4);  unsqueeze_default = None
        reshape_default_1: "f32[1, 8192, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_1, _shape_param_1);  unsqueeze_default_1 = _shape_param_1 = None
        squeeze_dim: "f32[8192, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_1, 0);  reshape_default_1 = None
        unsqueeze_default_2: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg347_1, 3);  arg347_1 = None
        unsqueeze_default_3: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, 4);  unsqueeze_default_2 = None
        reshape_default_2: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_3, _shape_param_2);  unsqueeze_default_3 = _shape_param_2 = None
        squeeze_dim_1: "f32[1024, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_2, 0);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        unsqueeze_default_4: "f32[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(add_tensor_2, 3);  add_tensor_2 = None
        unsqueeze_default_5: "f32[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 4);  unsqueeze_default_4 = None
        reshape_default_3: "f32[1, 8192, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_5, _shape_param_3);  unsqueeze_default_5 = _shape_param_3 = None
        squeeze_dim_2: "f32[8192, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_3, 0);  reshape_default_3 = None
        unsqueeze_default_6: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg348_1, 3);  arg348_1 = None
        unsqueeze_default_7: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 4);  unsqueeze_default_6 = None
        reshape_default_4: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_7, _shape_param_4);  unsqueeze_default_7 = _shape_param_4 = None
        squeeze_dim_3: "f32[1024, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_4, 0);  reshape_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        unsqueeze_default_8: "f32[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(clone_2, 3);  clone_2 = None
        unsqueeze_default_9: "f32[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 4);  unsqueeze_default_8 = None
        reshape_default_5: "f32[1, 16384, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_9, _shape_param_5);  unsqueeze_default_9 = _shape_param_5 = None
        squeeze_dim_4: "f32[16384, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_5, 0);  reshape_default_5 = None
        unsqueeze_default_10: "f32[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg350_1, 3);  arg350_1 = None
        unsqueeze_default_11: "f32[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 4);  unsqueeze_default_10 = None
        reshape_default_6: "f32[1, 1024, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default_11, _shape_param_6);  unsqueeze_default_11 = _shape_param_6 = None
        squeeze_dim_5: "f32[1024, 1024]" = torch.ops.aten.squeeze.dim(reshape_default_6, 0);  reshape_default_6 = None
        return (squeeze_dim, squeeze_dim_1, squeeze_dim_2, squeeze_dim_3, squeeze_dim_4, squeeze_dim_5)

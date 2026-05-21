class GraphModule(torch.nn.Module):
    def forward(self, mm_default_216: "f32[8192, 1024]", inductor_seeds_default: "i64[99]", add_252: "f32[512, 16, 1024]", primals_355: "f32[1024]", primals_356: "f32[1024]", primals_357: "f32[4096, 1024]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        unsqueeze_default: "f32[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_default_216, 0);  mm_default_216 = None
        reshape_default: "f32[512, 16, 1, 1, 1024]" = torch.ops.aten.reshape.default(unsqueeze_default, _shape_param_0);  unsqueeze_default = _shape_param_0 = None
        permute_default: "f32[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(reshape_default, [0, 1, 4, 2, 3]);  reshape_default = None
        reshape_default_1: "f32[512, 16, 1024]" = torch.ops.aten.reshape.default(permute_default, _shape_param_1);  permute_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:147 in post_attention, code: attn_out = self.dropout(attn_out)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 95);  inductor_seeds_default = None
        inductor_random_default: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_scalar, reshape_default_1);  gt_scalar = reshape_default_1 = None
        mul_tensor_1: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_tensor: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_1, add_252);  mul_tensor_1 = add_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[512, 16, 1]" = var_mean_correction[0]
        getitem_1: "f32[512, 16, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        mul_tensor_2: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_3: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_2, primals_355);  mul_tensor_2 = primals_355 = None
        add_tensor_2: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_3, primals_356);  mul_tensor_3 = primals_356 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        reshape_default_2: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_2);  add_tensor_2 = _shape_param_2 = None
        permute_default_1: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_357, [1, 0]);  primals_357 = None
        return (reshape_default_2, permute_default_1)

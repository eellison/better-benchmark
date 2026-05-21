class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[256]", arg1_1: "f32[256]", arg2_1: "f32[8, 4096, 256]", arg3_1: "f32[512, 256]", arg4_1: "f32[512]", arg5_1: "f32[256, 512]", arg6_1: "f32[256]", arg7_1: "f32[8, 4096, 256]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1474 in forward_chunk, code: hidden_states = self.layer_norm(hidden_states)
        var_mean = torch.ops.aten.var_mean.correction(arg2_1, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 4096, 1]" = var_mean[0]
        getitem_1: "f32[8, 4096, 1]" = var_mean[1];  var_mean = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[2]" = torch.ops.prims.inductor_seeds.default(2, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1451 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default: "f32[8, 4096, 256]" = torch.ops.prims.inductor_random.default([8, 4096, 256], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        gt_1: "b8[8, 4096, 256]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.05);  inductor_random_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1437 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default_1: "f32[8, 4096, 512]" = torch.ops.prims.inductor_random.default([8, 4096, 512], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt: "b8[8, 4096, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_1, 0.05);  inductor_random_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1474 in forward_chunk, code: hidden_states = self.layer_norm(hidden_states)
        sub: "f32[8, 4096, 256]" = torch.ops.aten.sub.Tensor(arg2_1, getitem_1);  arg2_1 = getitem_1 = None
        add: "f32[8, 4096, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[8, 4096, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[8, 4096, 256]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[8, 4096, 256]" = torch.ops.aten.mul.Tensor(mul, arg0_1);  mul = arg0_1 = None
        add_1: "f32[8, 4096, 256]" = torch.ops.aten.add.Tensor(mul_1, arg1_1);  mul_1 = arg1_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1436 in forward, code: hidden_states = self.dense(hidden_states)
        view: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_1, [32768, 256]);  add_1 = None
        permute: "f32[256, 512]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        addmm: "f32[32768, 512]" = torch.ops.aten.addmm.default(arg4_1, view, permute);  arg4_1 = view = permute = None
        view_1: "f32[8, 4096, 512]" = torch.ops.aten.reshape.default(addmm, [8, 4096, 512]);  addmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1437 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        mul_2: "f32[8, 4096, 512]" = torch.ops.aten.mul.Tensor(gt, view_1);  gt = view_1 = None
        mul_3: "f32[8, 4096, 512]" = torch.ops.aten.mul.Tensor(mul_2, 1.0526315789473684);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1438 in forward, code: hidden_states = self.act_fn(hidden_states)
        relu: "f32[8, 4096, 512]" = torch.ops.aten.relu.default(mul_3);  mul_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1450 in forward, code: hidden_states = self.dense(hidden_states)
        view_2: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu, [32768, 512]);  relu = None
        permute_1: "f32[512, 256]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        addmm_1: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg6_1, view_2, permute_1);  arg6_1 = view_2 = permute_1 = None
        view_3: "f32[8, 4096, 256]" = torch.ops.aten.reshape.default(addmm_1, [8, 4096, 256]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1451 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        mul_4: "f32[8, 4096, 256]" = torch.ops.aten.mul.Tensor(gt_1, view_3);  gt_1 = view_3 = None
        mul_5: "f32[8, 4096, 256]" = torch.ops.aten.mul.Tensor(mul_4, 1.0526315789473684);  mul_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1567 in torch_dynamo_resume_in_forward_at_1565, code: hidden_states = hidden_states + self.feed_forward(attn_output)
        add_2: "f32[8, 4096, 256]" = torch.ops.aten.add.Tensor(arg7_1, mul_5);  arg7_1 = mul_5 = None
        return (add_2,)

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[8, 1024]", primals_5: "f32[768]", full_default: "i64[8, 1024]", add: "i64[8, 1024]", mul_2: "f32[8, 1024, 768]", gt: "b8[8, 1024, 768]", div: "f32[8, 1024, 1]", tangents_1: "f32[8, 1024, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:424 in forward, code: embeddings = self.dropout(embeddings)
        convert_element_type_3: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_6: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.0);  convert_element_type_3 = None
        mul_7: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(tangents_1, mul_6);  tangents_1 = mul_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:423 in forward, code: embeddings = self.LayerNorm(embeddings)
        mul_9: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_7, primals_5);  primals_5 = None
        mul_10: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_9, 768)
        sum_1: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_9, [2], True)
        mul_11: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_9, mul_2);  mul_9 = None
        sum_2: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_11, [2], True);  mul_11 = None
        mul_12: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2, sum_2);  sum_2 = None
        sub_3: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_10, sum_1);  mul_10 = sum_1 = None
        sub_4: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_3, mul_12);  sub_3 = mul_12 = None
        mul_13: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div, sub_4);  div = sub_4 = None
        mul_14: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_7, mul_2);  mul_2 = None
        sum_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_14, [0, 1]);  mul_14 = None
        sum_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_7, [0, 1]);  mul_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:420 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        full_default_2: "b8[8, 1024, 1]" = torch.ops.aten.full.default([8, 1024, 1], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_3: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[8, 1024, 768]" = torch.ops.aten.where.self(full_default_2, full_default_3, mul_13);  full_default_2 = None
        full_default_4: "f32[1, 768]" = torch.ops.aten.full.default([1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put: "f32[1, 768]" = torch.ops.aten.index_put.default(full_default_4, [full_default], where, True);  full_default_4 = full_default = where = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:419 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        eq_1: "b8[8, 1024]" = torch.ops.aten.eq.Scalar(add, 1)
        unsqueeze_3: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
        where_1: "f32[8, 1024, 768]" = torch.ops.aten.where.self(unsqueeze_3, full_default_3, mul_13);  unsqueeze_3 = None
        full_default_6: "f32[4098, 768]" = torch.ops.aten.full.default([4098, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_1: "f32[4098, 768]" = torch.ops.aten.index_put.default(full_default_6, [add], where_1, True);  full_default_6 = add = where_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:418 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        eq_2: "b8[8, 1024]" = torch.ops.aten.eq.Scalar(primals_1, 1)
        unsqueeze_4: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_2, -1);  eq_2 = None
        where_2: "f32[8, 1024, 768]" = torch.ops.aten.where.self(unsqueeze_4, full_default_3, mul_13);  unsqueeze_4 = full_default_3 = mul_13 = None
        full_default_8: "f32[50265, 768]" = torch.ops.aten.full.default([50265, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_2: "f32[50265, 768]" = torch.ops.aten.index_put.default(full_default_8, [primals_1], where_2, True);  full_default_8 = primals_1 = where_2 = None
        return (None, index_put_2, index_put_1, index_put, sum_3, sum_4)

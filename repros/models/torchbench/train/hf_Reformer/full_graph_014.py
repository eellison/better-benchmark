class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[8, 4096]", gt: "b8[8, 4096, 256]", lt: "b8[8, 64, 1, 1]", tangents_1: "f32[8, 4096, 256]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:249 in forward, code: position_encodings = torch.reshape(dropped_weights, (batch_size, sequence_length, -1))
        view_1: "f32[8, 64, 64, 256]" = torch.ops.aten.reshape.default(tangents_1, [8, 64, 64, 256])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:247 in forward, code: dropped_weights = dropped_transposed_weights.transpose(2, 1)
        permute_2: "f32[8, 64, 64, 256]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:244 in forward, code: dropped_transposed_weights = nn.functional.dropout2d(
        convert_element_type: "f32[8, 64, 1, 1]" = torch.ops.prims.convert_element_type.default(lt, torch.float32);  lt = None
        div: "f32[8, 64, 1, 1]" = torch.ops.aten.div.Scalar(convert_element_type, 0.95);  convert_element_type = None
        mul_3: "f32[8, 64, 64, 256]" = torch.ops.aten.mul.Tensor(permute_2, div);  permute_2 = div = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:242 in forward, code: transposed_weights = weights.transpose(2, 1)
        permute_3: "f32[8, 64, 64, 256]" = torch.ops.aten.permute.default(mul_3, [0, 2, 1, 3]);  mul_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:240 in forward, code: weights = torch.cat(broadcasted_weights, dim=-1)
        slice_1: "f32[8, 64, 64, 64]" = torch.ops.aten.slice.Tensor(permute_3, 3, 0, 64)
        slice_2: "f32[8, 64, 64, 192]" = torch.ops.aten.slice.Tensor(permute_3, 3, 64, 256);  permute_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:227 in forward, code: weight.expand((batch_size,) + self.axial_pos_shape + weight.shape[-1:]) for weight in self.weights
        sum_1: "f32[1, 1, 64, 192]" = torch.ops.aten.sum.dim_IntList(slice_2, [0, 1], True);  slice_2 = None
        view_2: "f32[1, 64, 192]" = torch.ops.aten.reshape.default(sum_1, [1, 64, 192]);  sum_1 = None
        sum_2: "f32[1, 64, 1, 64]" = torch.ops.aten.sum.dim_IntList(slice_1, [0, 2], True);  slice_1 = None
        view_3: "f32[64, 1, 64]" = torch.ops.aten.reshape.default(sum_2, [64, 1, 64]);  sum_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:339 in forward, code: embeddings = nn.functional.dropout(inputs_embeds, p=self.dropout, training=self.training)
        convert_element_type_1: "f32[8, 4096, 256]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_4: "f32[8, 4096, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.0526315789473684);  convert_element_type_1 = None
        mul_5: "f32[8, 4096, 256]" = torch.ops.aten.mul.Tensor(tangents_1, mul_4);  tangents_1 = mul_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:330 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        eq: "b8[8, 4096]" = torch.ops.aten.eq.Scalar(primals_1, -1)
        unsqueeze_1: "b8[8, 4096, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[8, 4096, 256]" = torch.ops.aten.where.self(unsqueeze_1, full_default, mul_5);  unsqueeze_1 = full_default = mul_5 = None
        full_default_1: "f32[320, 256]" = torch.ops.aten.full.default([320, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put: "f32[320, 256]" = torch.ops.aten.index_put.default(full_default_1, [primals_1], where, True);  full_default_1 = primals_1 = where = None
        return (None, index_put, view_3, view_2)

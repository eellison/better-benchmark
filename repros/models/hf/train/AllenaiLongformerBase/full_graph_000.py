import torch
import torch._inductor.inductor_prims  # registers inductor prims ops
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[8, 1024]", primals_2: "f32[50265, 768]", primals_3: "f32[4098, 768]", primals_4: "f32[1, 768]", primals_5: "f32[768]", primals_6: "f32[768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1486 in forward, code: token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        full_default: "i64[8, 1024]" = torch.ops.aten.full.default([8, 1024], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/modeling_utils.py:1010 in get_extended_attention_mask, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        full_default_1: "f32[8, 1, 1, 1024]" = torch.ops.aten.full.default([8, 1, 1, 1024], -0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1503 in forward, code: extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)[
        select: "f32[8, 1, 1024]" = torch.ops.aten.select.int(full_default_1, 1, 0);  full_default_1 = None
        select_1: "f32[8, 1024]" = torch.ops.aten.select.int(select, 1, 0);  select = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:378 in create_position_ids_from_input_ids, code: mask = input_ids.ne(padding_idx).int()
        ne: "b8[8, 1024]" = torch.ops.aten.ne.Scalar(primals_1, 1)
        convert_element_type: "i32[8, 1024]" = torch.ops.prims.convert_element_type.default(ne, torch.int32);  ne = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:379 in create_position_ids_from_input_ids, code: incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
        cumsum: "i64[8, 1024]" = torch.ops.aten.cumsum.default(convert_element_type, 1)
        convert_element_type_1: "i32[8, 1024]" = torch.ops.prims.convert_element_type.default(cumsum, torch.int32);  cumsum = None
        mul_1: "i32[8, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_1, convert_element_type);  convert_element_type_1 = convert_element_type = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:380 in create_position_ids_from_input_ids, code: return incremental_indices.long() + padding_idx
        convert_element_type_2: "i64[8, 1024]" = torch.ops.prims.convert_element_type.default(mul_1, torch.int64);  mul_1 = None
        add: "i64[8, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_2, 1);  convert_element_type_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:418 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding: "f32[8, 1024, 768]" = torch.ops.aten.embedding.default(primals_2, primals_1, 1);  primals_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:419 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_1: "f32[8, 1024, 768]" = torch.ops.aten.embedding.default(primals_3, add, 1);  primals_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:420 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embedding_2: "f32[8, 1024, 768]" = torch.ops.aten.embedding.default(primals_4, full_default);  primals_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:422 in forward, code: embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        add_1: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        add_2: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_1, embedding_2);  add_1 = embedding_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:423 in forward, code: embeddings = self.LayerNorm(embeddings)
        var_mean = torch.ops.aten.var_mean.correction(add_2, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 1024, 1]" = var_mean[0]
        getitem_1: "f32[8, 1024, 1]" = var_mean[1];  var_mean = None
        add_3: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        sub_1: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_2, getitem_1);  add_2 = getitem_1 = None
        mul_2: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = None
        mul_3: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2, primals_5)
        add_4: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_3, primals_6);  mul_3 = primals_6 = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[1]" = torch.ops.prims.inductor_seeds.default(1, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:424 in forward, code: embeddings = self.dropout(embeddings)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default([8, 1024, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default, 1e-30);  inductor_random_default = None
        mul_4: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt, add_4);  add_4 = None
        mul_5: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_4, 1.0);  mul_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:423 in forward, code: embeddings = self.LayerNorm(embeddings)
        div: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        return (mul_5, select_1, primals_1, primals_5, full_default, add, mul_2, gt, div)

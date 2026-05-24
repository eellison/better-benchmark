import torch
import torch._inductor.inductor_prims  # registers inductor prims ops
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[8, 4096]", primals_2: "f32[320, 256]", primals_3: "f32[64, 1, 64]", primals_4: "f32[1, 64, 192]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:330 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding: "f32[8, 4096, 256]" = torch.ops.aten.embedding.default(primals_2, primals_1);  primals_2 = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[2]" = torch.ops.prims.inductor_seeds.default(2, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:339 in forward, code: embeddings = nn.functional.dropout(inputs_embeds, p=self.dropout, training=self.training)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_1: "f32[8, 4096, 256]" = torch.ops.prims.inductor_random.default([8, 4096, 256], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt: "b8[8, 4096, 256]" = torch.ops.aten.gt.Scalar(inductor_random_default_1, 0.05);  inductor_random_default_1 = None
        mul: "f32[8, 4096, 256]" = torch.ops.aten.mul.Tensor(gt, embedding);  embedding = None
        mul_1: "f32[8, 4096, 256]" = torch.ops.aten.mul.Tensor(mul, 1.0526315789473684);  mul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:227 in forward, code: weight.expand((batch_size,) + self.axial_pos_shape + weight.shape[-1:]) for weight in self.weights
        expand_1: "f32[8, 64, 64, 64]" = torch.ops.aten.expand.default(primals_3, [8, 64, 64, 64]);  primals_3 = None
        expand_2: "f32[8, 64, 64, 192]" = torch.ops.aten.expand.default(primals_4, [8, 64, 64, 192]);  primals_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:240 in forward, code: weights = torch.cat(broadcasted_weights, dim=-1)
        cat: "f32[8, 64, 64, 256]" = torch.ops.aten.cat.default([expand_1, expand_2], -1);  expand_1 = expand_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:242 in forward, code: transposed_weights = weights.transpose(2, 1)
        permute: "f32[8, 64, 64, 256]" = torch.ops.aten.permute.default(cat, [0, 2, 1, 3]);  cat = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:244 in forward, code: dropped_transposed_weights = nn.functional.dropout2d(
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1);  inductor_seeds_default = None
        inductor_random_default: "f32[8, 64, 1, 1]" = torch.ops.prims.inductor_random.default([8, 64, 1, 1], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        lt: "b8[8, 64, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_default, 0.95);  inductor_random_default = None
        convert_element_type: "f32[8, 64, 1, 1]" = torch.ops.prims.convert_element_type.default(lt, torch.float32)
        div: "f32[8, 64, 1, 1]" = torch.ops.aten.div.Scalar(convert_element_type, 0.95);  convert_element_type = None
        mul_2: "f32[8, 64, 64, 256]" = torch.ops.aten.mul.Tensor(permute, div);  permute = div = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:247 in forward, code: dropped_weights = dropped_transposed_weights.transpose(2, 1)
        permute_1: "f32[8, 64, 64, 256]" = torch.ops.aten.permute.default(mul_2, [0, 2, 1, 3]);  mul_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:249 in forward, code: position_encodings = torch.reshape(dropped_weights, (batch_size, sequence_length, -1))
        view: "f32[8, 4096, 256]" = torch.ops.aten.reshape.default(permute_1, [8, 4096, -1]);  permute_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:343 in forward, code: embeddings = embeddings + position_embeddings
        add: "f32[8, 4096, 256]" = torch.ops.aten.add.Tensor(mul_1, view);  mul_1 = view = None
        return (add, primals_1, gt, lt)

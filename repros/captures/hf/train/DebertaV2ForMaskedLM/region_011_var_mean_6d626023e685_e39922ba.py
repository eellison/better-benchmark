"""
Standalone repro captured via capture_hook.
Label: hf_DebertaV2ForMaskedLM_train
Pattern hash: 6d626023e685
Shape hash: e39922ba
"""
_shapes_config = "(T([128100, 1536], f32), T([8, 512], i64), T([512, 1536], f32), T([1, 512], i64), T([1536], f32), T([1536], f32), T([1536, 1536], f32), T([1536, 1536], f32), S([4096, 1536]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, primals_3: "f32[128100, 1536]", primals_1: "i64[8, 512]", primals_4: "f32[512, 1536]", primals_2: "i64[1, 512]", primals_5: "f32[1536]", primals_6: "f32[1536]", primals_7: "f32[1536, 1536]", primals_9: "f32[1536, 1536]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:535 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding_default: "f32[8, 512, 1536]" = torch.ops.aten.embedding.default(primals_3, primals_1, 0);  primals_3 = primals_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:538 in forward, code: position_embeddings = self.position_embeddings(position_ids.long())
        embedding_default_1: "f32[1, 512, 1536]" = torch.ops.aten.embedding.default(primals_4, primals_2);  primals_4 = primals_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:544 in forward, code: embeddings = embeddings + position_embeddings
        add_tensor: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(embedding_default, embedding_default_1);  embedding_default = embedding_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:552 in forward, code: embeddings = self.LayerNorm(embeddings)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[8, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-07);  getitem = None
        rsqrt_default: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        mul_tensor: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_tensor, primals_5);  mul_tensor = primals_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:561 in forward, code: embeddings = embeddings * mask
        add_tensor_2: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_tensor_1, primals_6);  mul_tensor_1 = primals_6 = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[73]" = torch.ops.prims.inductor_seeds.default(73, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:563 in forward, code: embeddings = self.dropout(embeddings)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default([8, 512, 1536], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor_2: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_scalar, add_tensor_2);  gt_scalar = add_tensor_2 = None
        mul_tensor_3: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 1.1111111111111112);  mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        reshape_default: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_tensor_3, _shape_param_0);  mul_tensor_3 = _shape_param_0 = None
        permute_default: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_default_1: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
        return (reshape_default, permute_default, permute_default_1)



def make_inputs():
    return [
    torch.randn([128100, 1536], dtype=torch.float32, device='cuda'),
    torch.randint(0, 128100, [8, 512], dtype=torch.int64, device='cuda'),
    torch.randn([512, 1536], dtype=torch.float32, device='cuda'),
    torch.randint(0, 512, [1, 512], dtype=torch.int64, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 1536], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 1536], dtype=torch.float32, device='cuda'),
    [4096, 1536],  # _shape_param_0
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")

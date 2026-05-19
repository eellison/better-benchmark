"""
Standalone repro captured via capture_hook.
Label: hf_GoogleFnet_train
Pattern hash: 166d931cff4f
Shape hash: 432a2049
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, primals_2: "i64[1, 512]", primals_4: "f32[32000, 768]", primals_1: "i64[32, 512]", primals_5: "f32[4, 768]", primals_6: "f32[512, 768]", primals_3: "i64[1, 512]", primals_7: "f32[768]", primals_8: "f32[768]", primals_9: "f32[768, 768]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:479 in forward, code: buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
        expand_default: "i64[32, 512]" = torch.ops.aten.expand.default(primals_2, _shape_param_0);  primals_2 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:129 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding_default: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(primals_4, primals_1, 3);  primals_4 = primals_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:130 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embedding_default_1: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(primals_5, expand_default);  primals_5 = expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:132 in forward, code: embeddings = inputs_embeds + token_type_embeddings
        add_tensor: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(embedding_default, embedding_default_1);  embedding_default = embedding_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:134 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_default_2: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(primals_6, primals_3);  primals_6 = primals_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:135 in forward, code: embeddings += position_embeddings
        add_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor, embedding_default_2);  add_tensor = embedding_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:136 in forward, code: embeddings = self.LayerNorm(embeddings)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_1, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[32, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_2: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_tensor_1, getitem_1);  add_tensor_1 = getitem_1 = None
        mul_tensor: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, primals_7);  mul_tensor = primals_7 = None
        add_tensor_3: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, primals_8);  mul_tensor_1 = primals_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/fnet/modeling_fnet.py:137 in forward, code: embeddings = self.projection(embeddings)
        reshape_default: "f32[16384, 768]" = torch.ops.aten.reshape.default(add_tensor_3, _shape_param_1);  add_tensor_3 = _shape_param_1 = None
        permute_default: "f32[768, 768]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
        return (reshape_default, permute_default)


def _default_make_inputs():
    return [
    torch.randint(0, 4, [1, 512], dtype=torch.int64, device='cuda'),
    torch.randn([32000, 768], dtype=torch.float32, device='cuda'),
    torch.randint(0, 32000, [32, 512], dtype=torch.int64, device='cuda'),
    torch.randn([4, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randint(0, 512, [1, 512], dtype=torch.int64, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    [32, 512],  # _shape_param_0
    [16384, 768],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
